from urllib.parse import urlparse


def uri_validator(url):
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except:
        return False
