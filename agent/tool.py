from enum import Enum
from typing import Type, Optional, Literal
from string import Template
from pandas import DataFrame

from homeharvest import scrape_property

from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.tools import BaseTool

# Debug
from langchain.globals import set_verbose, set_debug

from agent.listingscraper.parse import (
    expand_to_csv_row,
    format_human_readable,
    save_row_data,
)


class ListingType(Enum):
    SOLD = "SOLD"
    FOR_SALE = "FOR_SALE"
    FOR_RENT = "FOR_RENT"
    PENDING = "PENDING"


class HomeSearchResultsInput(BaseModel):
    """Input for home search library"""

    location: str = Field(description="house location to search")
    listing_type: ListingType = Field(
        default=ListingType.FOR_SALE, description="type of listing"
    )
    radius: Optional[float] = None


# This subclasses langchain's BaseTool to create a custom Tool to pass into OpenAI.
# https://python.langchain.com/v0.1/docs/modules/tools/custom_tools/
# Using a @tool decorator can work as well. NOTE (btamayo): I'm not sure if using the Subclass also needs the docstring.
# The 'properties' returned are the following schema: https://github.com/Bunsly/HomeHarvest?tab=readme-ov-file#property-schema
class HomeSearchResultsTool(BaseTool):
    """Tool that queries several real state marketplaces for listings for homes on sale. It includes parameters such as location, which is passed in to the search as a string, a type of listing, which defaults to FOR_SALE to indicate searching for listings that are currently for sale, and radius, which defaults to 5.0.

    Property
    ├── Basic Information:
    │ ├── property_url
    │ ├── mls
    │ ├── mls_id
    │ └── status

    ├── Address Details:
    │ ├── street
    │ ├── unit
    │ ├── city
    │ ├── state
    │ └── zip_code

    ├── Property Description:
    │ ├── style
    │ ├── beds
    │ ├── full_baths
    │ ├── half_baths
    │ ├── sqft
    │ ├── year_built
    │ ├── stories
    │ └── lot_sqft

    ├── Property Listing Details:
    │ ├── days_on_mls
    │ ├── list_price
    │ ├── list_date
    │ ├── pending_date
    │ ├── sold_price
    │ ├── last_sold_date
    │ ├── price_per_sqft
    │ ├── parking_garage
    │ └── hoa_fee

    ├── Location Details:
    │ ├── latitude
    │ ├── longitude
    │ ├── nearby_schools


    ├── Agent Info:
    │ ├── agent
    │ ├── agent_email
    │ └── agent_phone

    ├── Broker Info:
    │ ├── broker
    │ ├── broker_email
    │ └── broker_website
    """

    name: str = "home_search_results_tool"
    description: str = (
        "Uses homeharvest as a result fetching tool for the real state market"
    )
    max_results: int = 5
    args_schema: Type[BaseModel] = HomeSearchResultsInput

    def _run(
        self,
        location: str,
        listing_type: Optional[ListingType] = "FOR_SALE",
        radius: Optional[float] = 5.0,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        try:
            if listing_type is None:
                listing_type = ListingType.FOR_SALE

            # TODO (btamayo): Validate the inputs here so that we have proper use of API calls. Sometimes 
            # it does not "correctly" use or infer the "right" inputs.

            properties = scrape_property(
                location=location, listing_type="FOR_SALE", radius=radius
            )

            print(properties)

            properties_expanded = []
            for _, row in properties.iloc[0 : self.max_results].iterrows():
                save_row_data(row)
                expanded_row = format_human_readable(row)
                properties_expanded.append(expanded_row)
        except Exception as e:
            return str(e)

        return "\n".join(properties_expanded)
