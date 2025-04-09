from pynamodb.models import Model
from pynamodb.attributes import UnicodeAttribute



class SynapseEdgeResultsModel(Model):
    """
    A DynamoDB store for results
    """
    class Meta:
        table_name = "CloudomeResults"
        region = "us-east-1"

    graph_id = UnicodeAttribute(hash_key=True)
    synapse_id = UnicodeAttribute(range_key=True) # "syn_x100_y20_z42_pre150_post60"

class ContactEdgeResultsModel(Model):
    """
    A DynamoDB store for results
    """
    class Meta:
        table_name = "CloudomeResults"
        region = "us-east-1"

    graph_id = UnicodeAttribute(hash_key=True)
    synapse_id = UnicodeAttribute(range_key=True) # "ctc_x100_y20_z42_pre150_post60_w100"