#!/bin/sh

# Decrypt the file
mkdir $HOME/tests
# --batch to prevent interactive command
# --yes to assume "yes" for questions
gpgtar --decrypt --directory . --gpg-args="--passphrase=$EXAMPLE_XODR_PASSKEY --batch --quiet --yes" ./example_networks.gpg 