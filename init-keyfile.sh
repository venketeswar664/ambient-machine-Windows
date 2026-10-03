
#!/bin/bash
set -e

echo "Setting up MongoDB keyfile..."
echo "${MONGO_REPLICA_SET_KEY:-myReplicaKey}" > /data/configdb/keyfile
chmod 600 /data/configdb/keyfile
chown mongodb:mongodb /data/configdb/keyfile
echo "Keyfile setup complete."


