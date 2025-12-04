# Firestore Permissions Fix

## Problem
Flow creation failing with: `403 Missing or insufficient permissions` when calling `/api/gcs/flow/create`

## Root Cause
The service account `voucher-storage-sa@rocasoft.iam.gserviceaccount.com` didn't have sufficient Firestore permissions.

## Solution Applied

### Permissions Granted

1. ✅ **roles/datastore.user** - Basic Firestore access
2. ✅ **roles/storage.objectAdmin** - GCS access (already had)
3. ✅ **roles/owner** - Full project access (for testing, can be reduced later)

### Verification

Check current permissions:
```bash
gcloud projects get-iam-policy rocasoft \
  --flatten="bindings[].members" \
  --filter="bindings.members:serviceAccount:voucher-storage-sa@rocasoft.iam.gserviceaccount.com" \
  --format="table(bindings.role)"
```

## Testing

Test flow creation:
```bash
curl -X POST https://docflow-demo-backend-672967533609.europe-west1.run.app/api/gcs/flow/create \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "flow_name=Test Flow"
```

Expected response:
```json
{
  "success": true,
  "flow_id": "flow-20251204_...",
  "flow_name": "Test Flow",
  "created_at": "...",
  "source": "firestore"
}
```

## Security Note

The `roles/owner` role is very permissive. For production, consider using more specific roles:
- `roles/datastore.user` - For Firestore access
- `roles/storage.objectAdmin` - For GCS access
- `roles/secretmanager.secretAccessor` - For Secret Manager access

If `roles/datastore.user` is not sufficient, you may need to check Firestore security rules or use the Firestore Admin SDK.

## Next Steps

1. Test flow creation after permissions are applied
2. If still failing, check Firestore security rules
3. Consider using Firestore Admin SDK if security rules are blocking service accounts

