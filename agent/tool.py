from enum import Enum
import logging
import os
from pathlib import Path
from typing import Type, Optional, Literal
from string import Template
from pandas import DataFrame

from homeharvest import scrape_property

from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.tools import BaseTool


human_readable_row_tpl = Template(
    """
---
$mls_id: $street, $city, $zip_code

property style: $style
street: $street
city: $city
zip Code: $zip_code
bedrooms: $beds bedrooms
stories: $stories stories
full baths: $full_baths
half baths: $half_baths
sqft: $sqft sqft
listed price: $list_price
sold for: $sold_price
nearby schools: $nearby_schools
listing url: $property_url

description: $text
---
"""
)


def format_human_readable(property: DataFrame) -> str:
    return human_readable_row_tpl.substitute(property.to_dict())


def save_rows_to_csv(properties: DataFrame, filename: str = "listings.csv") -> str:
    print(f"Saving rows data to {filename}...")

    # todo: update path access to be cleaner
    data_base_dir_path = Path(os.getcwd(), "data")
    listings_csv_path = Path(data_base_dir_path, filename)

    if not data_base_dir_path.exists():
        data_base_dir_path.mkdir()

    if not listings_csv_path.exists():
        listings_csv_path.touch()

    properties.to_csv(listings_csv_path)


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
    radius: Optional[float] = Field(
        description="""
        Radius in miles to find comparable properties based on individual addresses. Example: 5.5 (fetches properties within a 5.5-mile radius if location is set."""
    )

    # Everything below is a param to the Tool, but not to the homeharvest scraper
    bedroom_number: Optional[int] = Field(
        description="""The number of bedrooms a user is looking for in a property. If not provided, it defaults to 2.0.
    """
    )
    bathroom_number: Optional[float] = Field(
        description="""The number of bathrooms a user is looking for in a property. If not provided, it defaults to 2.0."""
    )
    min_price: Optional[int] = Field(
        description="""The minimum price of a property to search for in United States Dollars. If not provided, it defaults to 10000000"""
    )
    max_price: Optional[int] = Field(
        description="""The maximum price of a property to search for in United States Dollars. If not provided, it defaults to 100000000"""
    )


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

    max_results: int = 20
    args_schema: Type[BaseModel] = HomeSearchResultsInput

    def _run(
        self,
        location: str,
        listing_type: Optional[ListingType] = "FOR_SALE",
        # todo: what happens if we input a float?
        min_price: Optional[int] = 10000000,
        max_price: Optional[int] = 100000000,
        bedroom_number: Optional[int] = 2.0,
        bathroom_number: Optional[float] = 2.0,
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
            save_rows_to_csv(properties)

            logging.info("Inferred location for location: %s", location)
            logging.info("Inferred min_price for filtering: %s", min_price)
            logging.info("Inferred max_price for filtering: %s", max_price)
            logging.info("Inferred bedroom_number for filtering: %s", bedroom_number)
            logging.info("Inferred bathroom_number for filtering: %s", bathroom_number)
            logging.info("Inferred listing_type for filtering: %s", listing_type)
            logging.info("Inferred radius for radius: %s", radius)

            res_df = properties[properties["list_price"] <= max_price]
            res_df = res_df[res_df["list_price"] >= min_price]
            res_df = res_df[res_df["beds"] >= bedroom_number]

            # not sure how the LLM is going to parse this, but we'll see
            res_df["bathrooms"] = res_df["full_baths"] + (res_df["half_baths"] * 0.5)
            res_df = res_df[res_df["bathrooms"] >= bathroom_number]

            properties_expanded = []
            for _, row in res_df.iloc[0 : self.max_results].iterrows():
                expanded_row = format_human_readable(row)
                properties_expanded.append(expanded_row)
        except Exception as e:
            return str(e)

        return "\n".join(properties_expanded)
