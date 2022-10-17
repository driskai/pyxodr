#!/bin/sh

# --batch to prevent interactive command
# --yes to assume "yes" for questions
gpgtar --decrypt --directory ./tests/ --gpg-args="--passphrase=$EXAMPLE_XODR_PASSKEY --batch --quiet --yes" ./tests/example_networks.gpg 