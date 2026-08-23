# Pareto fronts

These functions expose the full candidate set a detector selects its final partition from, letting you inspect or re-select solutions yourself.

Seven of the eleven detector entry points have one. `gdpso` and `cdrme` optimize a single scalar, so they have no Pareto front and no `gdpso_fronts` / `cdrme_fronts`; `mocd_q` and `mocd_d` are multi-objective but expose no front accessor.

::: pymocd.smocc_fronts

::: pymocd.mopots_fronts

::: pymocd.mopots_ladder

::: pymocd.hpmocd_fronts

::: pymocd.mmcomo_fronts

::: pymocd.ccm_fronts

::: pymocd.krm_fronts

::: pymocd.moga_net_fronts
