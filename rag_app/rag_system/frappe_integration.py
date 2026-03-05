"""
Frappe Integration for RAG System
==================================

Usage in Frappe:
API Method: rag_app.rag_system.frappe_integration.query_employee

Example from Python controller:
    import frappe
    result = frappe.call('rag_app.rag_system.frappe_integration.query_employee', question="What is John's salary?")

Example from JavaScript:
    frappe.call({
        method: 'rag_app.rag_system.frappe_integration.query_employee',
        args: { question: 'What is John salary?' },
        callback: function(r) {
            console.log(r.message);
        }
    });
"""

import frappe
from pathlib import Path
import sys
import importlib

# Add rag_system to path
rag_system_path = Path(__file__).resolve().parent
if str(rag_system_path) not in sys.path:
    sys.path.insert(0, str(rag_system_path))

@frappe.whitelist(allow_guest=True)
def query_employee(question: str):
    """
    Query employee database using SQL or RAG
    
    Args:
        question (str): Natural language question
    
    Returns:
        dict: {
            "answer": str,
            "query_type": "SQL" or "RAG",
            "sql": str (if SQL query),
            "raw_data": list (if SQL query),
            "matched_docs": list (if RAG query)
        }
    """
    try:
        try:
            from .query_llm_doctype import ask
        except ImportError:
            ask = importlib.import_module("query_llm_doctype").ask

        # Force refresh so latest DocType rows/fields are reflected in SQL + RAG paths.
        result = ask(question, force_init=True)
        return result
    except Exception as e:
        frappe.log_error(f"RAG Query Error: {str(e)}", "RAG System")
        return {
            "error": str(e),
            "answer": f"Sorry, I encountered an error: {str(e)}"
        }

@frappe.whitelist(allow_guest=True)
def get_query_stats()-> dict:
    """Get statistics about the employee database"""
    try:
        from .doctype_data_loader import get_employee_count
    except ImportError:
        get_employee_count = importlib.import_module("doctype_data_loader").get_employee_count
    
    try:
        return {
            "total_employees": get_employee_count(),
            "status": "connected",
            "data_source": "Frappe DocType (Employee Data)",
        }
    except Exception as e:
        frappe.log_error(f"RAG Stats Error: {str(e)}", "RAG System")
        return {
            "error": str(e)
        }
