import pandas as pd
import numpy as np
import sys
from keys import CRFM_API_KEY
from api_access import APIAccess

sys.path.append('/Users/khanda/Documents/code/projects/pulls/GitHub/benchmarking')

from src.common.authentication import Authentication
from src.common.perspective_api_request import PerspectiveAPIRequest, PerspectiveAPIRequestResult
from src.common.request import Request, RequestResult
from src.common.tokenization_request import TokenizationRequest, TokenizationRequestResult
from src.proxy.accounts import Account
from proxy.remote_service import RemoteService

class CRFMAccess(APIAccess):
    def request(self, model, format, needs_instruction):
        """
        Query the API with the generated prompt and retrieve an output of the probabilities of each token
        
        Parameters:
            model (str): the model on CRFM to query with generated prompt
            format (str): the format of the prompt ['arrow', 'qa']
            needs_instruction (bool): True if need to include instruction in prompt and False otherwise
        Returns:
            request_result (RequestResult): output from CRFM API query
        """

        auth = Authentication(api_key=CRFM_API_KEY)
        prompt = self.generate_formatted_prompt(format, needs_instruction, to_togethercomputer=False)
        service = RemoteService("https://crfm-models.stanford.edu")
        request = Request(
            prompt=prompt, 
            model="ai21/j1-jumbo",
            top_k_per_token=4,
            max_tokens=0,
            echo_prompt=True,
        )
        
        request_result: RequestResult = service.make_request(auth, request)
        
        return request_result
    
    def to_numpy_dataframe(self, output):
        """
        Reformat the output of the API into a numpy dataframe
            
        Args:
            output (RequestResult): output from CRFM API query
        Returns:
            unpacked_df (pd.DataFrame): the DataFrame obtained from the API call
        """
        unpacked_df = pd.DataFrame(output.completions[0].tokens)
        
        unpacked_df["%"] = unpacked_df["logprob"].apply(lambda x: 100*np.e**x)
        unpacked_df.rename(columns={'logprob':'token_logprobs', 'text':'tokens'}, inplace=True)
        
        return unpacked_df
    
    