"""
Category mapping utilities for backend to frontend category translation
Maps to exactly 4 categories: SPA, Invoices, ID, Proof of Payment
"""

def map_backend_to_ui_category(backend_classification: str) -> str:
    """
    Map backend classification to frontend UI category (4 categories only)
    
    Args:
        backend_classification: Backend classification (e.g., 'MPU', 'Invoice', 'Contract', 'SPA', etc.)
        
    Returns:
        UI category string: 'SPA', 'Invoices', 'ID', or 'Proof of Payment'
    """
    if not backend_classification:
        return 'SPA'  # Default to SPA for unknown
    
    classification = backend_classification.strip().upper()
    
    # Direct matches for the 4 categories
    if classification == 'SPA':
        return 'SPA'
    if classification == 'INVOICES' or classification == 'INVOICE':
        return 'Invoices'
    if classification == 'ID':
        return 'ID'
    if classification == 'PROOF OF PAYMENT' or classification == 'PROOFOFPAYMENT':
        return 'Proof of Payment'
    
    # ID/Passport types - map to 'ID'
    id_keywords = ['ID', 'PASSPORT', 'IDENTIFICATION', 'DRIVER', 'LICENSE', 'PASSPORT']
    if any(keyword in classification for keyword in id_keywords):
        return 'ID'
    
    # Invoice types - map to 'Invoices'
    if 'INVOICE' in classification:
        return 'Invoices'
    
    # Proof of Payment types - map to 'Proof of Payment'
    payment_keywords = ['PAYMENT', 'RECEIPT', 'BANK', 'TRANSFER', 'PROOF', 'CONFIRMATION', 'VOUCHER']
    if any(keyword in classification for keyword in payment_keywords):
        # Check if it's a payment receipt/confirmation, not an invoice
        if 'INVOICE' not in classification:
            return 'Proof of Payment'
    
    # SPA types (Sales, Purchases, Contract) - map to 'SPA'
    spa_keywords = [
        'SPA', 'SALES', 'PURCHASE', 'CONTRACT', 'AGREEMENT', 
        'BROKER', 'PROPERTY MANAGEMENT', 'RENEWAL', 'REFUND', 
        'CANCELLATION', 'TENANCY', 'LEASE', 'MPU', 'MPV', 'MSL', 
        'MRT', 'MJV', 'VOUCHER'
    ]
    if any(keyword in classification for keyword in spa_keywords):
        return 'SPA'
    
    # Default fallback to SPA (most common category)
    return 'SPA'

