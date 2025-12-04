# GCS to Firestore Migration Guide

## Overview

This migration script syncs all existing flows and files from GCS to Firestore, ensuring fast indexed queries while maintaining GCS for file storage.

## Prerequisites

1. Firestore service must be configured and running
2. GCS credentials must be available
3. Backend virtual environment activated

## Usage

### Test Migration (Dry Run)

Preview what will be migrated without making changes:

```bash
cd backend
python scripts/migrate_gcs_to_firestore.py --dry-run
```

### Migrate All Flows

Migrate all flows and their files from GCS to Firestore:

```bash
cd backend
python scripts/migrate_gcs_to_firestore.py
```

### Migrate Specific Flow

Migrate only a specific flow:

```bash
cd backend
python scripts/migrate_gcs_to_firestore.py --flow-id flow-20251128_121507
```

## What Gets Migrated

### Flow Metadata
- Flow ID
- Flow name
- Created date
- File count
- Status
- Source

### File Metadata
- Document ID (generated)
- Filename
- Flow ID
- File size
- GCS path (`gcs_path` or `gcs_temp_path`)
- Organized path (if processed)
- Processing status
- Migration timestamp

## What Does NOT Get Migrated

- Actual file content (stays in GCS)
- GCS bucket structure (unchanged)
- Existing Firestore records (skipped)

## Migration Process

1. **Fetch flows from GCS** - Lists all flow metadata files
2. **Check Firestore** - For each flow, checks if already exists
3. **Create flow records** - Creates missing flows in Firestore
4. **Fetch files** - Gets all files (temp, organized, failed) for each flow
5. **Check existing docs** - Queries Firestore documents for the flow
6. **Create file records** - Creates Firestore documents for missing files
7. **Report results** - Shows summary of migrated items

## Output Example

```
============================================================
🚀 Starting GCS to Firestore migration
============================================================

📂 Fetching flows from GCS...
✅ Found 15 flows in GCS

[1/15] Processing flow: flow-20251128_121507
  ✓ Flow already exists in Firestore: Invoice Processing
  📁 Checking files for flow: Invoice Processing
  📊 Found 25 files in GCS (temp: 0, organized: 23, failed: 2)
  📚 Found 25 existing documents in Firestore

[2/15] Processing flow: flow-20251127_093042
  ✅ Created flow in Firestore: Receipt Scanning
  📁 Checking files for flow: Receipt Scanning
  📊 Found 12 files in GCS (temp: 5, organized: 7, failed: 0)
  📚 Found 0 existing documents in Firestore
  ✅ Migrated 12 file records to Firestore

...

============================================================
📊 Migration Summary
============================================================
Flows checked:    15
Flows created:    3
Flows existing:   12
Files checked:    234
Files created:    87
Files existing:   147
Errors:           0
============================================================
```

## Safety Features

### Idempotent
- Safe to run multiple times
- Skips existing records
- No duplicate data created

### Non-Destructive
- Only creates new records
- Never deletes or modifies GCS data
- Never overwrites Firestore records

### Error Handling
- Continues on individual errors
- Logs all errors
- Provides summary at end

## After Migration

### Verify Results

Check Firestore console:
```
https://console.firebase.google.com/project/rocasoft/firestore/data
```

### Test API Endpoints

Test flow listing:
```bash
curl http://localhost:8000/api/gcs/flows
```

Should return `"source": "firestore"` in response

### Verify Performance

Compare query times before/after migration:
- Flow listing should be 10-50x faster
- File listings should be 10x faster

## Troubleshooting

### "Firestore service not available"
- Check `FIRESTORE_PROJECT_ID` in .env
- Verify `GOOGLE_APPLICATION_CREDENTIALS` path
- Ensure Firestore API is enabled

### "GCS client not initialized"
- Check `GCS_BUCKET_NAME` in .env
- Verify GCS credentials
- Ensure bucket exists

### Migration takes too long
- Migrate specific flows first: `--flow-id <flow_id>`
- Check network connectivity
- Verify Firestore quotas not exceeded

### Duplicate documents
- Should not happen (script checks existing)
- If occurs, check `document_id` generation logic
- Firestore documents are keyed by unique document_id

## Rollback

If needed:
1. Migration is additive only
2. GCS data remains unchanged
3. Can delete Firestore records if needed
4. Backend automatically falls back to GCS if Firestore unavailable

## Next Steps

After migration:
1. Test API endpoints
2. Verify data in Firestore console
3. Monitor performance improvements
4. Check logs for any errors or warnings
5. Run verify_optimizations.py to confirm improvements

---

**Status**: Ready for use  
**Last Updated**: November 28, 2025

