from peewee import *
import datetime
import racemachine.config as config

db = PostgresqlDatabase(
        config.get('database.name'),
        user=config.get('database.username'),
        password=config.get('database.password')
        )

class BaseModel(Model):
    class Meta:
        database = db

class Face(BaseModel):
    recognizer_label = IntegerField(null=True)
    race = FixedCharField(null=True)
    sex = FixedCharField(null=True)
    images_collected = IntegerField(default=1)
    created_at = DateTimeField(default=datetime.datetime.now)
