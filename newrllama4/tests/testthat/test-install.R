test_that("installation functions work", {
  
  # Test lib_is_installed function
  expect_type(lib_is_installed(), "logical")
  
  # Test get_lib_path function
  lib_path <- get_lib_path()
  expect_type(lib_path, "character")
  expect_true(length(lib_path) > 0)
  
  # Test that lib path is in user directory
  expect_true(grepl("newrllama4", lib_path))
})

test_that("installation parameters are valid", {
  
  # Test that we can access internal installation variables
  # This tests the install.R logic without actually downloading
  expect_silent({
    # These should not throw errors
    lib_path <- get_lib_path()
    is_installed <- lib_is_installed()
  })
  
  # Test that installation directory can be created
  temp_dir <- tempdir()
  expect_true(dir.exists(temp_dir))
})