"""Data-dependent tests read the ecr database (ECR_DB_URL); there is no file fallback.

Import this before anything loads data. It pins ECR_AS_OF to the window the golden fingerprints and
workbook expectations were built on, so later rows in a growing database cannot move them. Spawned
workers and the notebook kernel inherit the setting.
"""
import os

AS_OF = '2026-03-31'
os.environ['ECR_AS_OF'] = AS_OF  # overrides a shell value: the fixtures are built on this window
