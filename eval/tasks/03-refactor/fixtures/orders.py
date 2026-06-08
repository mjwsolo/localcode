def process_order(order):
    if not order.get("items"):
        raise ValueError("order must have items")
    if not order.get("customer"):
        raise ValueError("order must have customer")
    for item in order["items"]:
        if item.get("qty", 0) <= 0:
            raise ValueError(f"item {item.get('sku')} has invalid qty")

    total = 0
    for item in order["items"]:
        total += item["price"] * item["qty"]
    if order.get("discount"):
        total = total * (1 - order["discount"])

    lines = [f"Receipt for {order['customer']}:"]
    for item in order["items"]:
        lines.append(f"  {item['sku']} x{item['qty']} @ {item['price']} = {item['price'] * item['qty']}")
    lines.append(f"TOTAL: {total:.2f}")
    return "\n".join(lines)
