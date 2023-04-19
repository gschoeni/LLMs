
import os
import requests

class Branch(object):
    """An Oxen branch."""

    def __init__(self, name: str, commit_id: str):
        """Create a new repository.

        :param name: The name to the branch.
        :param commit_id: The commit id of the branch.
        """
        
        self.name = name
        self.commit_id = commit_id
    
    def from_data(data: dict):
        return Branch(data['name'], data['commit_id'])