# Copyright (c) 2026, Frappe Technologies and contributors
# For license information, please see license.txt

import frappe
from frappe.model.document import Document


class EmployeeData(Document):
	def before_save(self):
		"""Generate full name before saving"""
		if self.first_name and self.last_name:
			self.full_name = f"{self.first_name} {self.last_name}"
	
	def validate(self):
		"""Validate employee data"""
		# Ensure annual salary matches monthly salary * 12
		if self.monthly_salary and not self.annual_salary:
			self.annual_salary = self.monthly_salary * 12
