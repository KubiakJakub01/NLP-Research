import re


def extract_invoice_details(text: str) -> dict[str, str]:
    """
    Extract invoice details from raw text.

    Args:
        text: Raw invoice text

    Returns:
        Dictionary with extracted invoice_number, due_date, and total_amount
    """
    if not text or not text.strip():
        return {}

    result = {}

    # Extract invoice number
    invoice_pattern = r'Invoice Number:\s*([A-Z0-9-]+)'
    invoice_match = re.search(invoice_pattern, text)
    if invoice_match:
        result['invoice_number'] = invoice_match.group(1)

    # Extract due date
    due_date_pattern = r'Due Date:\s*(\d{4}-\d{2}-\d{2})'
    due_date_match = re.search(due_date_pattern, text)
    if due_date_match:
        result['due_date'] = due_date_match.group(1)

    # Extract total amount (remove $ and get numeric part)
    total_pattern = r'Total Amount:\s*\$([0-9.]+)'
    total_match = re.search(total_pattern, text)
    if total_match:
        result['total_amount'] = total_match.group(1)

    return result


# Test cases
def test_extract_invoice_details():
    """Test the invoice extraction function with various cases."""

    # Test case 1: Standard invoice
    test_text_1 = """
    INVOICE
    Billed to: John Doe
    Invoice Number: INV-2024-00123
    Date of Issue: 2024-07-15
    Due Date: 2024-08-14
    Item Description      Quantity      Price      Total
    ----------------------------------------------------
    Product A                  2        $50.00     $100.00
    Service B                  1       $150.00     $150.00
    ----------------------------------------------------
    Subtotal: $250.00
    Tax (10%): $25.00
    Total Amount: $275.00
    """

    expected_1 = {
        'invoice_number': 'INV-2024-00123',
        'due_date': '2024-08-14',
        'total_amount': '275.00',
    }

    result_1 = extract_invoice_details(test_text_1)
    assert result_1 == expected_1, f'Test 1 failed: {result_1} != {expected_1}'
    print('Test 1 passed: Standard invoice')

    # Test case 2: Missing fields
    test_text_2 = """
    INVOICE
    Billed to: Jane Smith
    Invoice Number: INV-2024-00456
    Date of Issue: 2024-07-20
    Total Amount: $150.00
    """

    expected_2 = {'invoice_number': 'INV-2024-00456', 'total_amount': '150.00'}

    result_2 = extract_invoice_details(test_text_2)
    assert result_2 == expected_2, f'Test 2 failed: {result_2} != {expected_2}'
    print('Test 2 passed: Missing fields')

    print('All tests passed!')


def main():
    raw_text = """
    INVOICE
    Billed to: John Doe
    Invoice Number: INV-2024-00123
    Date of Issue: 2024-07-15
    Due Date: 2024-08-14
    Item Description      Quantity      Price      Total
    ----------------------------------------------------
    Product A                  2        $50.00     $100.00
    Service B                  1       $150.00     $150.00
    ----------------------------------------------------
    Subtotal: $250.00
    Tax (10%): $25.00
    Total Amount: $275.00
    """

    result = extract_invoice_details(raw_text)
    print('Extracted invoice details:')
    print(result)

    # Run test cases
    print('\nRunning test cases...')
    test_extract_invoice_details()


if __name__ == '__main__':
    main()
