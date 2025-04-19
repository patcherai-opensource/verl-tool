#!/usr/bin/env python
import json
import requests
import fire
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def _send_request(url, trajectory_id, action):
    """
    Internal helper function to send a single request to the server and log the response.
    Returns True if successful, otherwise False.
    """
    payload = {
        "trajectory_ids": [
                trajectory_id + '84f8dd9d-40a1-4f8f-aeb3-a614aeb41203',
                trajectory_id + 'd18aeeee-96d2-45a4-b7a4-2c3feedffae4'
            ],
        "actions": [action, action],
        'finish': [True, True],
        'extra_fields': [
            {
                'golden_answers': ['Natalia Gastiain Tena'],
                'gt': 'Natalia Gastiain Tena',
                'id': 41214,
                'index': 41214,
                'question': 'who plays the wildling woman in game of thrones',
                'split': 'train',
                # 'url': 'http://localhost:22015/wikipedia_en_all_maxi_2022-05/A/User:The_other_Kiwix_guy/Landing',
                'url': 'https://tigerai.ca/wiki/wikipedia_en_all_maxi_2022-05/A/User:The_other_Kiwix_guy/Landing'
            },
            {
                'golden_answers': ['Natalia Gastiain Tena'],
                'gt': 'Natalia Gastiain Tena',
                'id': 41214,
                'index': 41214,
                'question': 'who plays the wildling woman in game of thrones',
                'split': 'train',
                # 'url': 'http://localhost:22015/wikipedia_en_all_maxi_2022-05/A/User:The_other_Kiwix_guy/Landing',
                'url': 'https://tigerai.ca/wiki/wikipedia_en_all_maxi_2022-05/A/User:The_other_Kiwix_guy/Landing'
            }
        ],
        "finish": [False, False],
    }
    # payload = {'trajectory_ids': ['dc5c027a-85a3-4c7a-9076-d96341c6433c'], 'actions': ['<think>\nThe user wants to
    # know the number of La Liga goals scored by Cristiano Ronaldo in the current season. I am currently on a Wikipedia landing page. The best way to find this information is to search for "Cristiano Ronaldo" on Wikipedia. I will use the search bar, which has the ID `21`, to perform the search.\n</think>\n```type [21] [Cristiano Ronaldo] [1]```'], 'extra_fields': [{'question': 'how many laliga goals does ronaldo have this season', 'gt': '37', 'url': 'https://tigerai.ca/wiki/wikipedia_en_all_maxi_2022-05/A/User:The_other_Kiwix_guy/Landing'}]}

    logger.info(f"Sending request to {url}")
    logger.info(f"Payload: {json.dumps(payload, indent=2)}")

    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        result = response.json()
        logger.info(f"Response: {json.dumps(result, indent=2)}")

        if "observations" not in result:
            logger.error("Error: Response missing 'observations' field")
            return False

        observations = result["observations"]
        if not observations or not isinstance(observations, list):
            logger.error(f"Error: Expected observations to be a non-empty list, got {type(observations)}")
            return False

        logger.info("Test passed! ✅")
        logger.info(f"Received {len(observations)} observations:")
        for i, obs in enumerate(observations):
            logger.info(f"  Observation {i + 1}: {obs}")

        return True

    except requests.exceptions.RequestException as e:
        logger.error(f"Connection error: {e}")
        return False
    except json.JSONDecodeError:
        logger.error("Error: Invalid JSON response")
        return False
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return False


def test_connection(url="http://localhost:5000/get_observation", trajectory_id="test-trajectory-001"):
    """
    Test the connection to the tool server by sending multiple actions sequentially.
    """

    actions = [
        # "Invalid Action",
        # None,
        "<think>balabalabalabala</think>\n```type [16] [Cristiano Ronaldo] [1]```"
        # "<think>balabalabalabala</think>\n```click [99]```",
        # "<think>balabala</think>```type [1407] [death row inmates in the US] [1]```",
        # "<think>balabala</think>```scroll [down]```",
    ]

    overall_success = True
    for action in actions:
        success = _send_request(url, trajectory_id, action)
        if not success:
            overall_success = False

    return overall_success



def main():
    """
    Entry point for the test script.
    
    Start the Server:
    rm *.db* # Delete existed environment
    python -m verl_tool.servers.serve --tool_type text_browser --url=http://localhost:5000/get_observation

    Run with:
    python -m verl_tool.servers.tests.test_text_browser --url=http://localhost:5000/get_observation
    """
    fire.Fire(test_connection)


if __name__ == "__main__":
    main()