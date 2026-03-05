"""
DocType Data Loader for RAG System
===================================

Utilities to fetch employee data from Frappe DocType and convert it to:
1. Pandas DataFrame for SQL queries (in-memory SQLite)
2. Text documents for embeddings and RAG retrieval

This replaces the Excel file dependency with Frappe database.
"""

import frappe
import pandas as pd
from typing import List, Dict, Any


NON_DATA_FIELDTYPES = {
	"Section Break",
	"Column Break",
	"Tab Break",
	"Fold",
	"Button",
	"HTML",
	"Image",
	"Heading",
	"Table",
	"Table MultiSelect",
}


def get_employee_doctype_fields() -> List[str]:
	"""Return all data-bearing fieldnames from Employee Data DocType."""
	meta = frappe.get_meta("Employee Data")
	fieldnames = []

	for field in meta.fields:
		if not field.fieldname:
			continue
		if field.fieldtype in NON_DATA_FIELDTYPES:
			continue
		fieldnames.append(field.fieldname)

	# Ensure stable unique order.
	seen = set()
	unique_fields = []
	for fieldname in fieldnames:
		if fieldname not in seen:
			seen.add(fieldname)
			unique_fields.append(fieldname)

	return unique_fields


def get_all_employees() -> List[Dict[str, Any]]:
	"""
	Fetch all employee records from Employee Data DocType
	
	Returns:
		List of dictionaries containing employee data
	"""
	fields = get_employee_doctype_fields()
	order_by = "employee_number"
	if "employee_number" not in fields:
		order_by = "modified desc"

	employees = frappe.get_all(
		"Employee Data",
		fields=fields,
		order_by=order_by,
	)
	
	return employees


def get_employees_dataframe() -> pd.DataFrame:
	"""
	Get employee data as a Pandas DataFrame
	Suitable for loading into SQLite for analytical queries
	
	Returns:
		Pandas DataFrame with employee data
	"""
	employees = get_all_employees()
	
	if not employees:
		frappe.log_error("No employee data found in DocType", "RAG System")
		return pd.DataFrame()
	
	# Convert employees list to a clean dictionary format to avoid serialization issues
	clean_employees = []
	for emp in employees:
		clean_emp = {}
		for key, value in emp.items():
			# Convert any non-JSON-serializable objects to strings
			if value is None:
				clean_emp[key] = None
			elif isinstance(value, (str, int, float, bool)):
				clean_emp[key] = value
			elif hasattr(value, 'strftime'):  # datetime object
				clean_emp[key] = str(value.date()) if hasattr(value, 'date') else str(value)
			else:
				# For any other type, convert to string
				clean_emp[key] = str(value)
		clean_employees.append(clean_emp)
	
	df = pd.DataFrame(clean_employees)
	
	# Normalize column names for SQL (replace spaces with underscores)
	# Frappe DocType fieldnames are already SQL-friendly (snake_case).
	
	return df


def get_employee_field_labels() -> Dict[str, str]:
	"""Return mapping of fieldname -> label for Employee Data."""
	meta = frappe.get_meta("Employee Data")
	label_map: Dict[str, str] = {}
	for field in meta.fields:
		if field.fieldname:
			label_map[field.fieldname] = field.label or field.fieldname.replace("_", " ").title()
	return label_map


def employee_to_text(employee: Dict[str, Any]) -> str:
	"""
	Convert an employee record to text format for embeddings
	
	Args:
		employee: Dictionary with employee data
	
	Returns:
		Formatted text string representing the employee
	"""
	text_parts = []
	label_map = get_employee_field_labels()

	# Emit a normalized name line if first/last name fields exist.
	first_name = employee.get("first_name") or ""
	last_name = employee.get("last_name") or ""
	full_name = (f"{first_name} {last_name}").strip() or employee.get("full_name")
	if full_name:
		text_parts.append(f"Name: {full_name}")

	priority_fields = ["employee_number", "department", "monthly_salary", "annual_salary"]
	for key in priority_fields:
		if key in employee and employee.get(key) not in (None, ""):
			text_parts.append(f"{label_map.get(key, key)}: {employee.get(key)}")

	for key, value in employee.items():
		if key in priority_fields or key in {"first_name", "last_name", "full_name"}:
			continue
		if value in (None, ""):
			continue
		text_parts.append(f"{label_map.get(key, key)}: {value}")
	
	return "\n".join(text_parts)


def get_all_employee_texts() -> List[str]:
	"""
	Get all employee records as text documents for embeddings
	
	Returns:
		List of formatted text strings, one per employee
	"""
	employees = get_all_employees()
	texts = [employee_to_text(emp) for emp in employees]
	return texts


def get_employee_count() -> int:
	"""
	Get total count of employees in the system
	
	Returns:
		Count of employee records
	"""
	return frappe.db.count("Employee Data")


def get_employee_data_signature() -> Dict[str, Any]:
	"""
	Return a lightweight signature of Employee Data content/schema.
	Used to decide whether embeddings/index need regeneration.
	"""
	fields = get_employee_doctype_fields()
	latest_modified = frappe.db.sql(
		"""
		SELECT MAX(modified)
		FROM `tabEmployee Data`
		""",
		as_dict=False,
	)[0][0]

	return {
		"doctype": "Employee Data",
		"count": get_employee_count(),
		"fields": fields,
		"latest_modified": str(latest_modified) if latest_modified else None,
	}


def get_new_employees_since(last_embedded_ids: List[str]) -> List[Dict[str, Any]]:
	"""
	Get employee records that are NOT in last_embedded_ids list.
	Used for incremental embedding.
	
	Args:
		last_embedded_ids: List of employee_numbers that were already embedded
	
	Returns:
		List of new employee records to embed
	"""
	if not last_embedded_ids:
		return get_all_employees()
	
	fields = get_employee_doctype_fields()
	# Create placeholders for SQL IN clause
	placeholders = ','.join(['%s'] * len(last_embedded_ids))
	
	new_employees = frappe.db.sql(
		f"""
		SELECT {', '.join(fields)}
		FROM `tabEmployee Data`
		WHERE employee_number NOT IN ({placeholders})
		ORDER BY employee_number
		""",
		last_embedded_ids,
		as_dict=True
	)
	
	return new_employees if new_employees else []


def get_all_employee_numbers() -> List[str]:
	"""
	Get list of all employee_numbers in the DocType.
	Used to track which records are in FAISS index.
	"""
	results = frappe.db.sql(
		"SELECT employee_number FROM `tabEmployee Data` ORDER BY employee_number",
		as_dict=True
	)
	return [r['employee_number'] for r in results]


# Test functions
def test_data_loader():
	"""Test the data loader functions"""
	print("Testing Employee Data Loader")
	print("=" * 50)
	
	# Test count
	count = get_employee_count()
	print(f"\nTotal Employees: {count}")
	
	# Test DataFrame
	df = get_employees_dataframe()
	print(f"\nDataFrame shape: {df.shape}")
	print(f"Columns: {list(df.columns)}")
	print(f"\nFirst 3 rows:")
	print(df.head(3))
	
	# Test text conversion
	texts = get_all_employee_texts()
	print(f"\nGenerated {len(texts)} text documents")
	print(f"\nSample text document:")
	print("-" * 50)
	print(texts[0] if texts else "No data")
	print("-" * 50)
	
	return {
		"count": count,
		"dataframe_shape": df.shape,
		"texts_count": len(texts)
	}


if __name__ == "__main__":
	# For testing outside Frappe context
	print("This module must be run within Frappe context")
	print("Use: bench execute rag_app.rag_system.doctype_data_loader.test_data_loader")
