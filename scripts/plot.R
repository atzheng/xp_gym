x <- read_csv("output/limited_memory_output.csv")
ate = 0.3

x <- read_csv("output/ridesharing.csv")
ate_df = read_csv("~/research/network-dq/output/ate.csv") 
ate = (
  ate_df
  %>% mutate(delta=B-A)
  %>% summarise(ate=mean(delta))
) $ ate

summ = (
  x
  %>% gather(estimator, val, -steps, -env_id)
  # %>% filter(estimator == 'pricing_dn')
  %>% group_by(steps, estimator)
  %>% summarise(
        rmse=sqrt(mean((val - ate) ^ 2 / ate ^ 2)),
        # rmse_se=sqrt(sd((val - ate) ^ 2 / ate ^ 2) / sqrt(n())),
        bias=abs(mean(val - ate) / ate),
        sd=sd(val - ate) / ate,
      )
  %>% gather(var, val, -estimator, -steps)
)
(
summ %>% ggplot(aes(
        steps,
        val,
        ## ymin=rmse - 1.96 * rmse_se,
        ## ymax=rmse + 1.96 * rmse_se,
        color=estimator,
        fill=estimator
      ))
  ## + geom_ribbon(aes(color=NULL), alpha=0.2)
  + geom_line(alpha=0.8)
  + theme_minimal()
  + theme(legend.position="bottom")
  + facet_wrap(~ var, scales="free", label=label_both)
  # + scale_y_log10()
  + scale_y_continuous(breaks=seq(0, 1, by=0.1))
  + coord_cartesian(ylim=c(0.0, 1.0))
)
 
