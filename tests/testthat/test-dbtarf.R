test_that("dbtarf_train_predict returns finite predictions and an interval", {
  set.seed(1)
  n <- 120
  tr <- data.frame(y = rnorm(n), a = rnorm(n), b = rnorm(n))
  te <- tr[1:15, ]
  out <- dbtarf_train_predict(tr, te, "y", c("a", "b"),
                              coord_names = NULL, rf_tune = FALSE,
                              fair_profile_search = FALSE)
  expect_length(out$pred_test, nrow(te))
  expect_true(all(is.finite(out$pred_test)))
  expect_true(is.finite(out$pi_q90))
})
