from typing import Self
from pydantic import BaseModel, ConfigDict, Field
import pandas as pd
import datetime


class Workstation(BaseModel):
    name: str
    max_units_per_run: int
    minutes_per_run: float
    minutes_changeover_time_taste: int
    minutes_changeover_time_bottle_size: int
    starts_at: datetime.time
    stops_at: datetime.time


class Product(BaseModel):
    product_id: int
    product_name: str
    setting_taste: str
    setting_bottle_size: str | None
    workstation_type: str


class BillOfMaterial(BaseModel):
    parent_id: int
    parent_name: str
    component_id: int
    component_name: str
    component_quantity: float


class ProductionOrder(BaseModel):
    production_order_nr: str
    days_till_delivery: int
    product_id: int
    amount: int
    liters_required: float = Field(alias="liters required")
    minutes_bottling_time: float = Field(alias="Bottling time in minutes")


class Data(BaseModel):
    """Contains all the data needed for the production scheduling problem."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    workstations: list[Workstation]
    products: dict[int, Product]
    bill_of_materials: dict[int, BillOfMaterial]
    production_orders: list[ProductionOrder]
    workstation_df: pd.DataFrame
    products_df: pd.DataFrame
    bill_of_materials_df: pd.DataFrame
    production_orders_df: pd.DataFrame

    @classmethod
    def empty(cls) -> Self:
        return cls(
            workstations=[],
            products={},
            bill_of_materials={},
            production_orders=[],
            workstation_df=pd.DataFrame(),
            products_df=pd.DataFrame(),
            bill_of_materials_df=pd.DataFrame(),
            production_orders_df=pd.DataFrame(),
        )


def parse_data(path: str) -> Data:
    workstations_df = pd.read_excel(path, sheet_name="workstations").replace(
        pd.NA, None
    )
    products_df = pd.read_excel(path, sheet_name="products").replace(pd.NA, None)
    bill_of_materials_df = pd.read_excel(path, sheet_name="bill of materials").replace(
        pd.NA, None
    )
    production_orders_df = pd.read_excel(path, sheet_name="production orders").replace(
        pd.NA, None
    )

    workstations = [
        Workstation(**workstation)  # type: ignore
        for workstation in workstations_df.to_dict("records")
    ]
    products = {
        product["product_id"]: Product(**product)  # type: ignore
        for product in products_df.to_dict("records")
    }
    bill_of_materials = {
        bom["parent_id"]: BillOfMaterial(**bom)  # type: ignore
        for bom in bill_of_materials_df.to_dict("records")
    }
    production_orders = [
        ProductionOrder(**order)  # type: ignore
        for order in production_orders_df.to_dict("records")
    ]
    return Data(
        workstations=workstations,
        products=products,
        bill_of_materials=bill_of_materials,
        production_orders=production_orders,
        workstation_df=workstations_df,
        products_df=products_df,
        bill_of_materials_df=bill_of_materials_df,
        production_orders_df=production_orders_df,
    )


if __name__ == "__main__":
    data = parse_data("examples/data_v1.xlsx")
    print(data)
