# User Credentials

## Default Users

The following users are automatically created when the backend starts:

### 1. Admin User
- **Email:** `admin@example.com`
- **Password:** `Admin`
- **Role:** `admin`
- **Full Name:** Admin User

### 2. Agent User
- **Email:** `agent1@example.com`
- **Password:** `agent1`
- **Role:** `agent`
- **Full Name:** Agent 1

## Additional Default Admin

- **Email:** `admin@example.com`
- **Password:** `password`
- **Role:** `admin`
- **Full Name:** Admin User

## Usage

1. Start the backend server
2. Navigate to `/login` in the frontend
3. Use the credentials above to log in
4. Admin users can manage all agents and see all data
5. Agent users can only see their own data

## User Storage

Users are stored in `backend/users.json` with bcrypt-hashed passwords.

## Creating New Users

New users can be created by:
1. Using the registration form at `/login` (creates agent role by default)
2. Using the `POST /api/auth/register` endpoint
3. Admins can manage users via `GET /api/auth/users` and `PUT /api/auth/users/{id}/role`

