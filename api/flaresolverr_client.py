"""FlareSolverr client for bypassing Cloudflare protection."""

import requests
from typing import Optional
from dataclasses import dataclass


@dataclass
class FlareSolverResponse:
    """Response from FlareSolverr."""
    status: str
    solution_url: str
    solution_status: int
    solution_html: str
    cookies: list[dict]
    user_agent: str


class FlareSolverr:
    """Client for FlareSolverr proxy."""

    def __init__(self, url: str = "http://10.0.0.4:8191"):
        """Initialize FlareSolverr client.

        Args:
            url: FlareSolverr endpoint URL
        """
        self.url = url.rstrip("/")
        self.session = requests.Session()

    def get(self, url: str, timeout: int = 60000) -> Optional[FlareSolverResponse]:
        """Fetch a URL through FlareSolverr.

        Args:
            url: The URL to fetch
            timeout: Max wait time in ms (default 60s)

        Returns:
            FlareSolverResponse or None on error
        """
        try:
            response = self.session.post(
                f"{self.url}/v1",
                json={
                    "cmd": "request.get",
                    "url": url,
                    "maxTimeout": timeout,
                },
                timeout=timeout / 1000 + 10,  # Add buffer for network
            )
            response.raise_for_status()
            data = response.json()

            if data.get("status") != "ok":
                print(f"FlareSolverr error: {data.get('message', 'Unknown error')}")
                return None

            solution = data.get("solution", {})
            return FlareSolverResponse(
                status=data.get("status"),
                solution_url=solution.get("url", url),
                solution_status=solution.get("status", 0),
                solution_html=solution.get("response", ""),
                cookies=solution.get("cookies", []),
                user_agent=solution.get("userAgent", ""),
            )

        except requests.exceptions.RequestException as e:
            print(f"FlareSolverr request failed: {e}")
            return None

    def is_available(self) -> bool:
        """Check if FlareSolverr is running."""
        try:
            response = self.session.get(f"{self.url}/health", timeout=5)
            return response.status_code == 200
        except:
            return False


# Singleton instance
_client: Optional[FlareSolverr] = None


def get_client(url: str = "http://10.0.0.4:8191") -> FlareSolverr:
    """Get or create FlareSolverr client."""
    global _client
    if _client is None:
        _client = FlareSolverr(url)
    return _client


def fetch_with_flaresolverr(url: str, flaresolverr_url: str = "http://10.0.0.4:8191") -> Optional[str]:
    """Convenience function to fetch HTML through FlareSolverr.

    Args:
        url: URL to fetch
        flaresolverr_url: FlareSolverr endpoint

    Returns:
        HTML content or None on error
    """
    client = get_client(flaresolverr_url)
    response = client.get(url)
    if response and response.solution_status == 200:
        return response.solution_html
    return None


if __name__ == "__main__":
    # Test FlareSolverr
    client = FlareSolverr()

    print("Checking FlareSolverr availability...")
    if client.is_available():
        print("✓ FlareSolverr is running")

        # Test with a Cloudflare-protected site
        print("\nTesting Babepedia fetch...")
        response = client.get("https://www.babepedia.com/babe/Angela_White")

        if response:
            print(f"✓ Status: {response.solution_status}")
            print(f"✓ HTML length: {len(response.solution_html)} bytes")
            print(f"✓ User-Agent: {response.user_agent[:50]}...")

            # Quick image check
            import re
            pattern = r'href="(/pics/[^"]+\.jpg)"'
            matches = re.findall(pattern, response.solution_html)
            print(f"✓ Found {len(matches)} images")
        else:
            print("✗ Failed to fetch")
    else:
        print("✗ FlareSolverr is not available")
