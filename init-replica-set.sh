# #!/bin/bash

# echo "Waiting for MongoDB to start on 127.0.0.1:27018..."

# # Wait until MongoDB starts
# until mongosh --quiet --host 127.0.0.1 --port 27018 --eval "db.runCommand({ ping: 1 }).ok" | grep 1 &>/dev/null; do
#   sleep 1
# done

# echo "MongoDB has started successfully"

# echo "Initiating MongoDB replica set..."

# # Initiate the replica set
# mongosh -u root -p root --host 127.0.0.1 --port 27017 --eval "
#   rs.initiate({
#     _id: 'rs0',
#     members: [
#       {
#         _id: 0,
#         host: '127.0.0.1:27017'
#       }
#     ]
#   })

# "


#!/bin/bash
set -e

echo "Waiting for MongoDB to start on 127.0.0.1:27018..."
until mongosh --quiet --host 127.0.0.1 --port 27018 --eval "db.runCommand({ ping: 1 }).ok" | grep 1 &>/dev/null; do
  sleep 1
done

echo "MongoDB has started. Initiating replica set..."

mongosh -u root -p root --host 127.0.0.1 --port 27018 --authenticationDatabase admin --eval "
  rs.initiate({
    _id: 'rs0',
    members: [ { _id: 0, host: 'localhost:27018' } ]
  });

  db = db.getSiblingDB('sentinel_warehouse');
  db.createUser({
    user: 'sentinel',
    pwd: 'sentinelMongo_test123',
    roles: [ { role: 'readWrite', db: 'sentinel_warehouse' } ]
  });
"

echo "Replica set and user initialized successfully."
