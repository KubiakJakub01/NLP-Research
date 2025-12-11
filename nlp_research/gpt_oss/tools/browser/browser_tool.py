import contextvars
import dataclasses
import functools
import itertools
import json
import re
import textwrap
from collections.abc import AsyncIterator, Callable
from typing import Any, ParamSpec
from urllib.parse import quote

import pydantic
import structlog
import tiktoken
from openai_harmony import Message, TextContent

from .page_contents import Extract, PageContents

logger = structlog.stdlib.get_logger(component=__name__)


# TODO(zhuohan): Use the correct encoding at release
ENC_NAME = 'o200k_base'
FIND_PAGE_LINK_FORMAT = '# 【{idx}†{title}】'
PARTIAL_INITIAL_LINK_PATTERN = re.compile(r'^[^【】]*】')
PARTIAL_FINAL_LINK_PATTERN = re.compile(r'【\d*(?:†(?P<content>[^†】]*)(?:†[^†】]*)?)?$')
LINK_PATTERN = re.compile(r'【\d+†(?P<content>[^†】]+)(?:†[^†】]+)?】')

CITATION_OUTPUT_PATTERN = re.compile(r'【(?P<cursor>\d+)†(?P<content>[^†】]+)(?:†[^†】]+)?】')

CallParams = ParamSpec('CallParams')


_P = ParamSpec('_P')
_live_function_name = contextvars.ContextVar[str]('_live_function_name')


class ToolUsageError(Exception):
    pass


def function_the_model_can_call(
    fn: Callable[_P, AsyncIterator[Message]],
) -> Callable[_P, AsyncIterator[Message]]:
    fn.__fn_calling_tool_fn_type__ = 'function_the_model_can_call'  # type: ignore

    @functools.wraps(fn)
    async def inner(*args: _P.args, **kwargs: _P.kwargs) -> AsyncIterator[Message]:
        token = _live_function_name.set(fn.__name__)
        try:
            async for m in fn(*args, **kwargs):
                yield m
        finally:
            _live_function_name.reset(token)

    return inner


@functools.cache
def _tiktoken_vocabulary_lengths(enc_name: str) -> list[int]:
    encoding = tiktoken.get_encoding(enc_name)
    results = []
    for i in range(encoding.n_vocab):
        try:
            results.append(len(encoding.decode([i])))
        except Exception:  # pylint: disable=broad-exception-caught
            results.append(1)
    return results


@dataclasses.dataclass(frozen=True)
class Tokens:
    tokens: list[int]
    tok2idx: list[int]  # Offsets = running sum of lengths.


@functools.cache
def max_chars_per_token(enc_name: str) -> int:
    """Typical value is 128, but let's be safe."""
    tok_lens = _tiktoken_vocabulary_lengths(enc_name)
    return max(tok_lens)


def get_tokens(text: str, enc_name: str) -> Tokens:
    encoding = tiktoken.get_encoding(enc_name)
    tokens = encoding.encode(text, disallowed_special=())
    _vocabulary_lengths = _tiktoken_vocabulary_lengths(enc_name)
    tok2idx = [0] + list(itertools.accumulate(_vocabulary_lengths[i] for i in tokens))[:-1]
    result = Tokens(tokens=tokens, tok2idx=tok2idx)
    return result


def get_end_loc(
    loc: int,
    num_lines: int,
    total_lines: int,
    lines: list[str],
    view_tokens: int,
    encoding_name: str,
) -> int:
    if num_lines <= 0:
        # COMPUTE NUMBER OF LINES TO SHOW
        txt = join_lines(lines[loc:], add_line_numbers=True, offset=loc)
        # if the text is very short, no need to truncate at all
        # at least one char per token
        if len(txt) > view_tokens:
            # limit the amount of text we tokenize here
            upper_bound = max_chars_per_token(encoding_name)
            tok2idx = get_tokens(txt[: (view_tokens + 1) * upper_bound], encoding_name).tok2idx
            if len(tok2idx) > view_tokens:
                end_idx = tok2idx[view_tokens]
                num_lines = txt[:end_idx].count('\n') + 1  # round up
            else:
                num_lines = total_lines
        else:
            num_lines = total_lines

    return min(loc + num_lines, total_lines)


def get_page_metadata(
    curr_page: PageContents,
) -> dict[str, str | None | dict[str, str] | list[str]]:
    """Some attributes of the current page."""
    page_metadata: dict[str, str | None | dict[str, str] | list[str]] = {
        'url': curr_page.url,
        'title': curr_page.title,
    }
    return page_metadata


def join_lines(lines: list[str], add_line_numbers: bool = False, offset: int = 0) -> str:
    if add_line_numbers:
        return '\n'.join([f'L{i + offset}: {line}' for i, line in enumerate(lines)])
    return '\n'.join(lines)


def wrap_lines(text: str, width: int = 80) -> list[str]:
    lines = text.split('\n')
    wrapped = itertools.chain.from_iterable(
        (
            textwrap.wrap(line, width=width, replace_whitespace=False, drop_whitespace=False)
            if line
            else ['']
        )  # preserve empty lines
        for line in lines
    )
    return list(wrapped)


def strip_links(text: str) -> str:
    text = re.sub(PARTIAL_INITIAL_LINK_PATTERN, '', text)
    text = re.sub(PARTIAL_FINAL_LINK_PATTERN, lambda mo: mo.group('content'), text)
    text = re.sub(LINK_PATTERN, lambda mo: mo.group('content'), text)
    return text


def maybe_get_function_args(message: Message, tool_name: str = 'browser') -> dict[str, Any] | None:
    if not message.recipient.startswith(f'{tool_name}.'):
        return None

    contents = ''
    if len(message.content) == 1 and isinstance(message.content[0], TextContent):
        contents = message.content[0].text

    if not contents:
        return {}

    try:
        parsed_contents = json.loads(contents)
        if isinstance(parsed_contents, dict):
            return parsed_contents
    except json.JSONDecodeError:
        pass

    return None


async def run_find_in_page(
    pattern: str,
    page: PageContents,
    max_results: int = 50,
    num_show_lines: int = 4,
) -> PageContents:
    lines = wrap_lines(text=page.text)
    txt = join_lines(lines, add_line_numbers=False)
    without_links = strip_links(txt)
    lines = without_links.split('\n')

    result_chunks, snippets = [], []
    line_idx, match_idx = 0, 0
    while line_idx < len(lines):
        line = lines[line_idx]
        if pattern not in line.lower():
            line_idx += 1
            continue
        snippet = '\n'.join(lines[line_idx : line_idx + num_show_lines])
        link_title = FIND_PAGE_LINK_FORMAT.format(idx=f'{match_idx}', title=f'match at L{line_idx}')
        result_chunks.append(f'{link_title}\n{snippet}')
        snippets.append(
            Extract(url=page.url, text=snippet, title=f'#{match_idx}', line_idx=line_idx)
        )
        if len(result_chunks) == max_results:
            break
        match_idx += 1
        line_idx += num_show_lines

    urls = [page.url for _ in result_chunks]

    if result_chunks:
        display_text = '\n\n'.join(result_chunks)
    else:
        display_text = f'No `find` results for pattern: `{pattern}`'

    result_page = PageContents(
        url=f'{page.url}/find?pattern={quote(pattern)}',
        title=f'Find results for text: `{pattern}` in `{page.title}`',
        text=display_text,
        urls={str(i): url for i, url in enumerate(urls)},
        snippets={str(i): snip for i, snip in enumerate(snippets)},
    )
    return result_page


class SimpleBrowserState(pydantic.BaseModel):
    # maps page url to page contents
    pages: dict[str, PageContents] = pydantic.Field(default_factory=dict)
    # a sequential list of page urls
    page_stack: list[str] = pydantic.Field(default_factory=list)

    @property
    def current_cursor(self) -> int:
        return len(self.page_stack) - 1

    def add_page(self, page: PageContents) -> None:
        self.pages[page.url] = page
        self.page_stack.append(page.url)  # pylint: disable=no-member

    def get_page(self, cursor: int = -1) -> PageContents:
        if self.current_cursor < 0:
            raise ToolUsageError('No pages to access!')
        if cursor in (-1, self.current_cursor):
            return self.pages[self.page_stack[-1]]
        try:
            page_url = self.page_stack[cursor]
        except TypeError as e:
            raise ToolUsageError(
                f'`cursor` should be an integer, not `{type(cursor).__name__}`'
            ) from e
        except IndexError as e:
            raise ToolUsageError(
                f'Cursor `{cursor}` is out of range. '
                f'Available cursor indices: [0 - {self.current_cursor}].'
            ) from e
        return self.pages[page_url]

    def get_page_by_url(self, url: str) -> PageContents | None:
        if url in self.pages:
            return self.pages[url]
        return None

    def pop_page_stack(self) -> None:
        assert self.current_cursor >= 0, 'No page to pop!'
        self.page_stack.pop()  # pylint: disable=no-member
