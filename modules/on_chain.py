import requests

def get_on_chain_metrics(coin_id: str):
    """
    Fetch on-chain/community metrics from CoinGecko.
    Standardized to return a dictionary with consistent keys.
    """
    try:
        # Use lower-case coin_id as required by CoinGecko API
        url = f"https://api.coingecko.com/api/v3/coins/{coin_id.lower()}?localization=false&tickers=false&market_data=true&community_data=true&developer_data=true&sparkline=false"
        response = requests.get(url, timeout=10)
        
        if response.status_code != 200:
            return {'error': 'API Limit or Invalid ID'}
            
        data = response.json()
        market_data = data.get('market_data', {})
        community_data = data.get('community_data', {})
        developer_data = data.get('developer_data', {})

        return {
            'market_cap_rank': market_data.get('market_cap_rank', 'N/A'),
            'circulating_supply': market_data.get('circulating_supply', 0),
            'total_supply': market_data.get('total_supply', 0),
            'twitter_followers': community_data.get('twitter_followers', 0),
            'github_stars': developer_data.get('stars', 0),
        }
    except Exception:
        return {'error': 'Service Unavailable'}
