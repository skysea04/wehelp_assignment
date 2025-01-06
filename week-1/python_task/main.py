import csv
import json
import statistics
from urllib import request

PAGE_COUNT = 40
CATE_ID = "DSAA31"
PCHOME_URL = "https://ecshweb.pchome.com.tw/search/v4.3/all/results?cateid={cate_id}&attr=&pageCount={page_count}&page={page}"

TASK1_FILE = "products.txt"
TASK2_FILE = "best-products.txt"
TASK4_FILE = "standardization.csv"


def get_asus_laptop_data_in_page(page: int) -> tuple[int, list[dict]]:
    """I assume that the request will always be successful and the data will be valid."""
    with request.urlopen(
        PCHOME_URL.format(cate_id=CATE_ID, page_count=PAGE_COUNT, page=page)
    ) as response:
        data = json.loads(response.read().decode("utf-8"))

    return data["TotalPage"], data["Prods"]


def main():
    cur_page = 1
    total_page = float("inf")
    all_prods = []
    while cur_page <= total_page:
        total_page, prods = get_asus_laptop_data_in_page(cur_page)
        all_prods.extend(prods)
        cur_page += 1

    prices = [prod["OriginPrice"] for prod in all_prods]
    std_dev = statistics.stdev(prices)
    mean_price = statistics.mean(prices)
    i5_prices = []

    with (
        open(TASK1_FILE, "w") as f1,
        open(TASK2_FILE, "w") as f2,
        open(TASK4_FILE, "w") as f3,
    ):
        csv_writer = csv.writer(f3)

        f1.write("productID\n")
        f2.write("productID\n")
        csv_writer.writerow(["productID", "Price", "PriceZScore"])

        for prod in all_prods:
            product_id = prod["Id"]
            price = prod["OriginPrice"]
            price_z_score = round((price - mean_price) / std_dev, 6)

            f1.write(f"{product_id}\n")
            if (
                prod["reviewCount"]
                and prod["reviewCount"] >= 1
                and prod["ratingValue"]
                and prod["ratingValue"] > 4.9
            ):
                f2.write(f"{product_id}\n")

            csv_writer.writerow([product_id, price, price_z_score])

            if "i5處理器" in prod["Describe"]:
                i5_prices.append(price)

    print(statistics.mean(i5_prices))


if __name__ == "__main__":
    main()
