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
    """Input for home search library, subclass of BaseModel. This will be parsed by a parser and rendered into the LLM prompt in order to inform it how to use the HomeSearchResults tool. The default parser is https://api.python.langchain.com/en/latest/tools/langchain.tools.render.render_text_description_and_args.html"""

    location: str = Field(
        description="""The location to search for listings. This can either be a city, a specific address, a zip code, or a county name, or another location descriptive string. If it is not provided, use "San Jose". Some examples are:
            "San Jose" means "San Jose, California"
            "123 Main St, San Jose" means "123 Main St, San Jose, California"
            "95134" means the ZIP code "95134, United States of America"
            "London" means "London, United Kingdom"
            "Paris" means "Paris, France"
            "Bay Area" means "Bay Area, California"
        """
    )
    listing_type: ListingType = Field(
        default=ListingType.FOR_SALE,
        description="Type of listing. This is either FOR_SALE, SOLD, FOR_RENT, or PENDING. By default, the input is FOR_SALE.",
    )
    radius: Optional[float] = Field(
        description="""
        Radius in miles to find comparable properties based on individual addresses. Example: 5.5 (fetches properties within a 5.5-mile radius if location is set."""
    )

    # Everything below is a param to the Tool, but not to the homeharvest scraper
    bedroom_number: Optional[float] = Field(
        description="""The number of bedrooms a user is looking for in a property. If not provided, it defaults to 2.0.
    """
    )
    bathroom_number: Optional[float] = Field(
        description="""The number of bathrooms a user is looking for in a property. If not provided, it defaults to 2.0."""
    )
    min_price: Optional[float] = Field(
        description="""The minimum price of a property to search for. If not provided, it defaults to 1000000.0."""
    )
    max_price: Optional[float] = Field(
        description="""The maximum price of a property to search for. If not provided, it defaults to 10000000.0."""
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
    # https://api.python.langchain.com/en/latest/tools/langchain.tools.render.render_text_description_and_args.html
    description: str = (
        "Uses homeharvest API scraper to fetch results on properties currently for sale. Users can specify the location, min_price, max_price, bedroom_number, bathroom_number, listing_type, and radius. This tool runs a search and should only be used once in a chat or if the user requires a new search."
    )
    max_results: int = 5
    args_schema: Type[BaseModel] = HomeSearchResultsInput
    tags: list[str] = ["home", "real estate", "property", "search", "listing", "API"]

    def _run(
        self,
        location: str,
        min_price: Optional[float] = 1000000.0,
        max_price: Optional[float] = 10000000.0,
        bedroom_number: Optional[float] = 2.0,
        bathroom_number: Optional[float] = 2.0,
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

            print(f"Properties length: {len(properties)}")
            print(properties)

            # Debug
            logging.info("Inferred location for location: %s", location)
            logging.info("Inferred min_price for filtering: %s", min_price)
            logging.info("Inferred max_price for filtering: %s", max_price)
            logging.info("Inferred bedroom_number for filtering: %s", bedroom_number)
            logging.info("Inferred bathroom_number for filtering: %s", bathroom_number)
            logging.info("Inferred listing_type for filtering: %s", listing_type)
            logging.info("Inferred radius for radius: %s", radius)

            # TODO: Add programmatic filter [Jun 5, 2024, bianca-tamayo]
            print(f"Properties length: {len(properties)}")
            print(properties[0])

            properties_expanded = []
            for _, row in properties.iloc[0 : self.max_results].iterrows():
                # Format the row and add it to return
                expanded_row = format_human_readable(row)
                properties_expanded.append(expanded_row)

            # print(f"Saving rows...")
            # save_rows_to_csv(properties)
        except Exception as e:
            return str(e)

        return "\n".join(properties_expanded[0 : self.max_results])
