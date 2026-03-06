# Copyright (c) 2026, Frappe Technologies and Contributors
# See license.txt

import frappe
from frappe.tests.utils import FrappeTestCase


class TestEmployeeData(FrappeTestCase):
	def test_full_name_generation(self):
		"""Test that full name is generated correctly"""
		emp = frappe.get_doc({
			"doctype": "Employee Data",
			"employee_number": "TEST001",
			"first_name": "John",
			"last_name": "Doe",
			"gender": "Male",
			"monthly_salary": 5000
		})
		emp.insert()
		
		self.assertEqual(emp.full_name, "John Doe")
		emp.delete()
