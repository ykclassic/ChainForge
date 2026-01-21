# modules/on_chain.py (Minor update - return None instead of mixing types)
import requests

def get_on_chain_metrics(coin_id: str):
    """Fetch on-chain/community metrics from CoinGecko (free, no key needed)."""
    try:
        url = f"https://api.coingecko.com/api/v3/coins/{coin_id}?localization=false&tickers=false&market_data=false&community_data=true&developer_data=true&sparkline=false"
        data = requests.get(url, timeout=10).json()
        market_data = data.get('market_data', {})
        community_data = data.get('community_data', {})
        developer_data = data.get('developer_data', {})

        return {
            'market_cap_rank': market_data.get('market_cap_rank'),
            'circulating_supply': market_data.get('circulating_supply'),
            'total_supply': market_data.get('total_supply'),
            'twitter_followers': community_data.get('twitter_followers'),
            'github_stars': developer_data.get('stars'),
        }
    except Exception:
        return {'error': 'Unavailable'}
