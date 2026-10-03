import mongoengine as me
import datetime as dt


def utcnow():
    return dt.datetime.now(dt.timezone.utc)

class VideoBackup(me.Document):
    meta = {
        "collection": "video_backup",
        "indexes": [
            ("period_start", "-period_end"),
            "status",
            "azure_container",
            "azure_blob",
        ],
    }

    # What window did we back up?
    period_start = me.DateTimeField(required=True)
    period_end   = me.DateTimeField(required=True)
    timezone     = me.StringField(default="UTC")

    # What did we include?
    source_dirs  = me.ListField(me.StringField())
    file_count   = me.IntField(default=0)
    total_bytes  = me.LongField(default=0)

    # Artifact on disk
    archive_local_path = me.StringField()
    archive_size_bytes = me.LongField()
    archive_sha256     = me.StringField()
    archive_md5        = me.StringField()

    # Azure upload
    azure_container = me.StringField()
    azure_blob      = me.StringField()
    azure_url       = me.StringField()
    azure_etag      = me.StringField()
    azure_content_md5_b64 = me.StringField()  # what Azure reports
    validate_content = me.BooleanField(default=True)

    # Outcome
    status          = me.StringField(choices=("Pending", "Uploaded", "Skipped", "Failed"), default="Pending")
    integrity_ok    = me.BooleanField(default=False)
    error           = me.StringField()

    created_at      = me.DateTimeField(default=utcnow)
    updated_at      = me.DateTimeField(default=utcnow)

    def touch(self):
        self.updated_at = utcnow()