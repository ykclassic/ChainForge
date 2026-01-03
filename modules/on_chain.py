import requests

def get_on_chain_metrics(coin_id):
    """Fetch on-chain metrics from CoinGecko (free)."""
    try:
        data = requests.get(f"https://api.coingecko.com/api/v3/coins/{coin_id}").json()
        market_data = data.get('market_data', {})
        community_data = data.get('community_data', {})
        developer_data = data.get('developer_data', {})
        return {
            'market_cap_rank': market_data.get('market_cap_rank', 'N/A'),
            'circulating_supply': market_data.get('circulating_supply', 'N/A'),
            'total_supply': market_data.get('total_supply', 'N/A'),
            'twitter_followers': community_data.get('twitter_followers', 'N/A'),
            'github_stars': developer_data.get('stars', 'N/A'),
            'active_addresses': 'N/A'  # Placeholder for Dune/Glassnode in v0.6
        }
    except:
        return {'error': 'Unavailable'}
