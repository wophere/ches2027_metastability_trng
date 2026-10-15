PYTHON ?= python

.PHONY: reproduce verify
reproduce:
	$(PYTHON) reproduce.py

verify:
	$(PYTHON) verify.py
