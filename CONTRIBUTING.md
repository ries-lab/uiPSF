# Contributing

When contributing to this repository, please indicate the change you are making via issue or 
pull request.

## How to Work and Develop Together
Please follow good git practices for developing code. In particular that means:
1. Use one branch per feature that you develop (checkout from main branch as basis)
2. Merge main branch frequently (i.e. once a day) if the feature takes longer to develop, 
   otherwise you will get into merge hell.
3. After a feature is developed, please open a pull request from the feature branch to main and 
   describe the changes and let someone take a look.

## Pull Requests

1. Make sure that the `environment.yaml` is up-to-date, in case you have added other dependencies
2. Update the README.md if necessary
3. Remove all hard coded stuff in your code, in particular paths.

## Things that should not be pushed
- large files, binaries, images
- compiled code
- cache
