from pynamodb.models import Model
from pynamodb.attributes import UnicodeAttribute



class ResultsModel(Model):
    """
    A DynamoDB store for results
    """
    class Meta:
        table_name = "CloudomeResults"
        region = "us-east-1"

    graph_id = UnicodeAttribute(hash_key=True)
    synapse_id = UnicodeAttribute(range_key=True) # "pre150_post60_x100_y20_z42"
    # pre_id = UnicodeAttribute()
    # post_id = UnicodeAttribute()
    # centroid_xyz = UnicodeAttribute()

