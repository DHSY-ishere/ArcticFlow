package benchmark

import (
	"fmt"
	"strings"
)

// splitmix64 is a tiny deterministic hash used to synthesize token ids.
func splitmix64(x uint64) uint64 {
	x += 0x9E3779B97F4A7C15
	x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9
	x = (x ^ (x >> 27)) * 0x94D049BB133111EB
	return x ^ (x >> 31)
}

// SynthTokens returns n deterministic pseudo-token ids keyed by (ns, key).
func SynthTokens(ns, key uint64, n int) []uint32 {
	out := make([]uint32, n)
	seed := splitmix64(ns<<32 ^ key)
	for i := range out {
		out[i] = uint32(splitmix64(seed+uint64(i)) % 151643)
	}
	return out
}

// Questions are analytical questions the engine's planner resolves against GenSemanticYAML models.
var Questions = []string{
	"What is total revenue by region?",
	"top 5 customers by revenue",
	"monthly revenue in 2023",
	"average order value by sales channel for EMEA",
	"how many completed orders by product category",
	"units sold by category for Enterprise customers in 2024",
}

// FollowUps are context-dependent turns (inherit metric/filters from the previous turn).
var FollowUps = []string{
	"now break that down by segment",
	"what about 2023?",
	"only for APAC",
}

// GenSemanticYAML renders a Cortex-Analyst-style semantic model for a tenant. The model has the same
// logical schema as examples/semantic_models/sales_analytics.yaml (so the same questions resolve),
// padded with `fillerCols` extra documented dimension columns so the YAML reaches a realistic
// "large semantic model" size (~35 tokens per filler column).
func GenSemanticYAML(tenant, fillerCols int) string {
	p := fmt.Sprintf("t%02d_", tenant)
	var b strings.Builder
	w := func(f string, a ...any) { fmt.Fprintf(&b, f, a...) }
	w("name: tenant_%02d_sales\ndescription: Sales star schema for tenant %02d.\n\ntables:\n", tenant, tenant)
	w("  - name: %sorders\n    description: One row per order line.\n    primary_key: {columns: [order_id]}\n", p)
	w("    dimensions:\n      - {name: order_status, expr: status, data_type: TEXT, synonyms: [status], sample_values: [completed, pending, cancelled]}\n")
	w("      - {name: sales_channel, expr: channel, data_type: TEXT, synonyms: [channel], sample_values: [web, mobile, retail]}\n")
	perTable := fillerCols / 3
	for i := 0; i < perTable; i++ {
		w("      - name: attr_%03d\n        expr: attr_%03d\n        data_type: TEXT\n        description: Auxiliary attribute %d of the order line used for audit, reconciliation and downstream reporting.\n        sample_values: [alpha, beta, gamma, delta]\n", i, i, i)
	}
	w("    time_dimensions:\n      - {name: order_date, expr: order_date, data_type: DATE, synonyms: [date]}\n")
	w("    facts:\n      - {name: amount, expr: amount, data_type: NUMBER}\n      - {name: quantity, expr: quantity, data_type: NUMBER}\n")
	w("      - {name: customer_id, expr: customer_id, data_type: NUMBER}\n      - {name: product_id, expr: product_id, data_type: NUMBER}\n")
	w("    metrics:\n      - {name: total_revenue, expr: SUM(amount), synonyms: [revenue, sales]}\n")
	w("      - {name: units_sold, expr: SUM(quantity), synonyms: [units]}\n      - {name: order_count, expr: COUNT(order_id), synonyms: [orders]}\n")

	w("  - name: %scustomers\n    description: One row per customer.\n    primary_key: {columns: [customer_id]}\n    dimensions:\n", p)
	w("      - {name: customer_name, expr: customer_name, data_type: TEXT, synonyms: [customer, customers]}\n")
	w("      - {name: region, expr: region, data_type: TEXT, synonyms: [regions], sample_values: [EMEA, APAC, AMER, LATAM]}\n")
	w("      - {name: segment, expr: segment, data_type: TEXT, synonyms: [segments], sample_values: [Enterprise, SMB, Consumer]}\n")
	for i := 0; i < perTable; i++ {
		w("      - name: cust_attr_%03d\n        expr: cust_attr_%03d\n        data_type: TEXT\n        description: Customer master attribute %d maintained by the data governance team.\n        sample_values: [red, green, blue]\n", i, i, i)
	}

	w("  - name: %sproducts\n    description: One row per product.\n    primary_key: {columns: [product_id]}\n    dimensions:\n", p)
	w("      - {name: product_name, expr: product_name, data_type: TEXT, synonyms: [product, products]}\n")
	w("      - {name: category, expr: category, data_type: TEXT, synonyms: [categories], sample_values: [Hardware, Software, Services]}\n")
	for i := 0; i < fillerCols-2*perTable; i++ {
		w("      - name: prod_attr_%03d\n        expr: prod_attr_%03d\n        data_type: TEXT\n        description: Product catalogue attribute %d synchronised from the PIM system.\n        sample_values: [x, y, z]\n", i, i, i)
	}

	w("\nrelationships:\n")
	w("  - name: orders_to_customers\n    left_table: %sorders\n    right_table: %scustomers\n    relationship_columns:\n      - {left_column: customer_id, right_column: customer_id}\n", p, p)
	w("  - name: orders_to_products\n    left_table: %sorders\n    right_table: %sproducts\n    relationship_columns:\n      - {left_column: product_id, right_column: product_id}\n", p, p)
	w("\nmetrics:\n  - {name: average_order_value, expr: \"AVG(%sorders.amount)\", synonyms: [aov, average order value]}\n", p)
	return b.String()
}
