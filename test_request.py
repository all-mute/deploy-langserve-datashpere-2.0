import requests
import json

# В эти поля вам необходимо вставить свои данные
# Вмсето IAM токена, который действителен 12 часов, можно использовать статический API-ключ
# В этом случае замените заголовок запроса на "Authorization": "Api-key " + api_key
iam_token = "___"
folder_id = "___"
node_id = "___"
#alias_name = "datasphere.user.___"

base_url = "https://node-api.datasphere.yandexcloud.net"
url = f"{base_url}/invoke"

payload = {
    "input": "Привет. Как дела?"
}

default_headers={
    "x-node-id" : node_id,
    #"x-node-alias" : alias_name,
    "x-folder-id" : folder_id,
    "Authorization": "Bearer " + iam_token
}

response = requests.post(
        url=url,
        data=json.dumps(payload),
        headers=default_headers,
        timeout=300,
    )

print(response.json())