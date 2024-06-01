from enum import Enum
from typing import Type, Optional, Literal
from string import Template
from pandas import DataFrame

from homeharvest import scrape_property

from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.tools import BaseTool

template = Template("""
---
Property style: $style
Street: $street
City: $city
Zip Code: $zip_code
$beds beds
$stories stories
$full_baths full baths
$half_baths half baths
$sqft sqft
listed for $list_price
sold for $sold_price
nearby schools are $nearby_schools
listing url: $property_url

description: $text
---
""")

def expand_row(property: DataFrame) -> str:
    return template.substitute(property.to_dict())


class ListingType(Enum):
    SOLD = "SOLD"
    FOR_SALE = "FOR_SALE"
    FOR_RENT = "FOR_RENT"
    PENDING = "PENDING"

class HomeSearchResultsInput(BaseModel):
    """Input for home search library"""

    location: str = Field(description="house location to search")
    listing_type: ListingType = Field(
        default=ListingType.FOR_SALE,
        description="type of listing"
    )
    radius: Optional[float] = None


class HomeSearchResultsTool(BaseTool):
    """Tool that queries several real state marketplaces for on sale houses"""

    name: str = "home_search_results_tool"
    description: str = (
        "Uses homeharvest as a result fetching tool for the real state market"
    )
    max_results: int = 5
    args_schema: Type[BaseModel] = HomeSearchResultsInput

    def _run(
        self,
        location: str,
        listing_type: Optional[Literal["SOLD", "FOR_SALE", "FOR_RENT", "PENDING"]] = "FOR_SALE",
        radius: Optional[float] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        properties = scrape_property(
            location=location,
            listing_type=listing_type,
            radius=radius
        )

        properties_expanded = []
        for _, row in properties.iloc[0:self.max_results].iterrows():
            properties_expanded.append(expand_row(row))
        
        return "\n".join(properties_expanded)