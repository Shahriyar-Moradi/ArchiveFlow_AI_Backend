# Service Account Permissions Configuration

## Service Account
**Email:** `voucher-storage-sa@rocasoft.iam.gserviceaccount.com`

## Current Permissions (Admin Roles)

### Firestore/Datastore
- ✅ **roles/datastore.owner** - Full admin access to Firestore/Datastore
  - Create, read, update, delete documents
  - Manage indexes
  - Full administrative control

### Google Cloud Storage
- ✅ **roles/storage.admin** - Full admin access to GCS
  - Create, read, update, delete objects
  - Manage buckets
  - Generate signed URLs
  - Full administrative control

### Secret Manager
- ✅ **roles/secretmanager.secretAccessor** - Access to secrets
  - Read secret values
  - Access mounted secrets in Cloud Run

## Permission Summary

```bash
# View current permissions
gcloud projects get-iam-policy rocasoft \
  --flatten="bindings[].members" \
  --filter="bindings.members:serviceAccount:voucher-storage-sa@rocasoft.iam.gserviceaccount.com" \
  --format="table(bindings.role)"
```

**Current Roles:**
- `roles/datastore.owner` - Firestore admin
- `roles/storage.admin` - GCS admin
- `roles/secretmanager.secretAccessor` - Secret Manager access

## Why Admin Roles?

Admin roles provide:
- ✅ Full control over resources
- ✅ Ability to create, read, update, delete
- ✅ No permission-related errors
- ✅ Suitable for service accounts that need comprehensive access

## Security Considerations

While admin roles provide full access, they are appropriate for:
- Service accounts used by backend services
- Services that need to manage their own data
- Production workloads that require comprehensive access

**Best Practice:** These permissions are scoped to the service account, not the entire project. The service account can only access resources it's granted permissions to.

## Verification

Test that permissions are working:

```bash
# Test flow creation (uses Firestore)
curl -X POST https://docflow-demo-backend-672967533609.europe-west1.run.app/api/gcs/flow/create \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "flow_name=Test Flow"
```

Expected: `{"success": true, ...}`

## Updating Permissions

If you need to update permissions:

```bash
# Add a role
gcloud projects add-iam-policy-binding rocasoft \
  --member="serviceAccount:voucher-storage-sa@rocasoft.iam.gserviceaccount.com" \
  --role="ROLE_NAME"

# Remove a role
gcloud projects remove-iam-policy-binding rocasoft \
  --member="serviceAccount:voucher-storage-sa@rocasoft.iam.gserviceaccount.com" \
  --role="ROLE_NAME"
```

