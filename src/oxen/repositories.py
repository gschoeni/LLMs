
import os
import requests
from typing import List

from oxen.branches import Branch

class Repository(object):
    """An Oxen repository.

    This class is responsible for interacting with a remote Oxen repository.
    """

    def __init__(self, name: str, host: str = "hub.oxen.ai", api_key: str = None):
        """Create a new repository.

        :param name: The name to the repository. Format is :namespace/:name. For example ox/CatsVsDogs
        :param host: The host to the repository. Defaults to hub.oxen.ai.
        :param api_key: The API key to use for authentication. Defaults to the value of the OXEN_API_KEY environment variable.
        """
        self.name = name
        if not '/' in name:
            raise ValueError('Repository name must be in the format :namespace/:name. For example ox/CatsVsDogs')
        
        split = name.split('/')
        self.namespace = split[0]
        self.repo_name = split[1]
        self.host = host
        
        if api_key is None:
            self.api_key = os.environ.get('OXEN_API_KEY')
        else:
            self.api_key = api_key
        
        
    def _remote_url(self):
        return f'http://{self.host}/api/repos/{self.namespace}/{self.repo_name}'

    def _get_request(self, url: str) -> dict:
        headers = {'Authorization': f'Bearer {self.api_key}'}
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        return response.json()
    
    def _post_request(self, url: str, data: dict) -> dict:
        headers = {'Authorization': f'Bearer {self.api_key}', 'Content-Type': 'application/json', 'user-agent': 'Oxen/python-client'}
        print(f'POST {url} {data} {headers}')
        response = requests.post(url, data=data, headers=headers)
        response.raise_for_status()
        return response.json()

    def list_branches(self) -> List[Branch]:
        base_url = self._remote_url()
        url = f'{base_url}/branches'
        branches_json = self._get_request(url)
        if not 'branches' in branches_json:
            raise Exception(f'Invalid response from server: {branches_json}')
        
        for branch in branches_json['branches']:
            yield Branch(branch['name'], branch['commit_id'])

    def get_branch_by_name(self, name) -> Branch:
        base_url = self._remote_url()
        url = f'{base_url}/branches/{name}'
        branch_json = self._get_request(url)
        if not 'branch' in branch_json:
            raise Exception(f'Invalid response from server: {branch_json}')
        branch_json = branch_json['branch']
        return Branch(branch_json['name'], branch_json['commit_id'])

    def add_row(self, branch: Branch, file: str, data: dict):
        base_url = self._remote_url()
        # "/{namespace}/{repo_name}/staging/{identifier}/df/rows/{resource:.*}",
        url = f'{base_url}/staging/{self.namespace}/df/rows/{branch.name}/{file}'
        self._post_request(url, data)
        
        