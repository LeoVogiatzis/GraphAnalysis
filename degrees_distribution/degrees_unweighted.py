try:
    from degrees import *
except ImportError:
    from .degrees import *


def main():
    graph = read_from_neo4j_database()
    print("Read from database")
    attacks = count_active_relationships_for_user("ATTACKS", graph)
    trades = count_active_relationships_for_user("TRADES", graph)
    messages = count_active_relationships_for_user("messages", graph)
    attacks_p_df = count_passive_relationships_for_user("ATTACKS", graph)
    trades_p_df = count_passive_relationships_for_user("TRADES", graph)
    messages_p_df = count_passive_relationships_for_user("messages", graph)

    total_attacks = count_pas_active_relations("ATTACKS", graph)
    total_trades = count_pas_active_relations("TRADES", graph)
    total_messages = count_pas_active_relations("messages", graph)

    print('Calculate total_interactions_per_user')
    active_sums, passive_sums, max_total_attacks, min_total_attacks, avg_total_attacks, max_total_trades, \
    min_total_trades, avg_total_trades, max_total_messages, min_total_messages, avg_total_messages = \
        total_interactions_per_user(attacks, trades, messages, attacks_p_df, trades_p_df, messages_p_df,
                                    total_attacks, total_trades, total_messages)
    print('Calculate min/max/average for all the cases..')

    max_attacks, max_trades, max_messages, max_attacks_p_df, max_trades_p_df, max_messages_p_df = \
        max_metrics_for_users(attacks, trades, messages, attacks_p_df, trades_p_df, messages_p_df)

    min_attacks, min_trades, min_messages, min_attacks_p_df, min_trades_p_df, min_messages_p_df = \
        min_metrics_for_users(attacks, trades, messages, attacks_p_df, trades_p_df, messages_p_df)

    average_at, average_tr, average_mes, average_pas_at, average_pas_tr, average_pas_mes = \
        average_metrics_for_users(attacks, trades, messages, attacks_p_df, trades_p_df, messages_p_df)

    max_sum, min_sum, average_sum, max_sum_passive, min_sum_passive, average_sum_passive = \
        act_pass_metrics(active_sums, passive_sums)
    max_attacks, min_attacks, average_attacks, max_trades, min_trades, average_trades, max_messages, min_messages, \
    average_messages = in_out_degree(total_attacks, total_trades, total_messages)

    plots(attacks, trades, messages, attacks_p_df, trades_p_df, messages_p_df, active_sums, passive_sums, total_attacks,
          total_trades, total_messages)

    plots_for_min_max_avg(max_attacks, max_trades, max_messages,
                          max_attacks_p_df, max_trades_p_df, max_messages_p_df,
                          min_attacks, min_trades, min_messages,
                          min_attacks_p_df, min_trades_p_df, min_messages_p_df,
                          average_at, average_tr, average_mes,
                          average_pas_at, average_pas_tr, average_pas_mes,
                          max_sum, min_sum, average_sum, max_sum_passive,
                          min_sum_passive, average_sum_passive,
                          max_total_attacks, min_total_attacks, avg_total_attacks,
                          max_total_trades, min_total_trades, avg_total_trades,
                          max_total_messages, min_total_messages, avg_total_messages)

    jac_simmilarity(total_attacks, total_trades, total_messages)

    x = 1


if __name__ == '__main__':
    main()
