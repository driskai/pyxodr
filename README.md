# pyxodr

![Tests](https://github.com/driskai/pyxodr/actions/workflows/tests.yml/badge.svg)
![Python 3.7+](https://img.shields.io/badge/python-3.7+-brightgreen)

Read OpenDRIVE files into a class structure that represents road objects as arrays of `(x,y,z)` coordinates rather than parameterised functions.

This class structure implements an API which should act as a middle layer between OpenDRIVE files and other applications, used to create road networks from their coordinates.

<p align="center">
<img src="https://raw.githubusercontent.com/driskai/pyxodr/main/docs/source/Ex_LHT-Complex-X-Junction_old_spec.png" width="30%" />
&nbsp; &nbsp; &nbsp; &nbsp;
<img src="https://raw.githubusercontent.com/driskai/pyxodr/main/docs/source/UC_2Lane-RoundAbout-3Arms.png" width="30%" />
&nbsp; &nbsp; &nbsp; &nbsp;
<img src="https://raw.githubusercontent.com/driskai/pyxodr/main/docs/source/UC_Motorway-Exit-Entry-DirectJunction.png" width="30%" />
</p>

## Installation
Install with `pip`:
```bash
pip install git+https://github.com/driskai/pyxodr
```
To install extra requirements for developing:
```bash
pip install "pyxodr[dev] @ git+https://github.com/driskai/pyxodr"
```

## Testing

Testing is done on the OpenDRIVE example files. I have not included them in this repository as ASAM requires you enter your details to access them, so I assume they don't want them publically distributed through any means other than their own website. You can access them [here](https://www.asam.net/standards/detail/opendrive/).

You can also test on the networks for the OpenSCENARIO example files, obtainable [here](https://www.asam.net/standards/detail/openscenario/).

Once you've downloaded these files, create an `example_networks` subdirectory under `tests` and place them there.
```bash
.
├── docs
├── pyxodr
│   ├── geometries
│   ├── road_objects
│   └── utils
└── tests
    ├── example_networks
    │   ├── Ex_Lane-Border
    │   │   ...
    │   └── UC_Simple-X-Junction-TrafficLights
    ├── output_plots
```
and then you should be able to run the (Pytest) tests as normal:
```bash
pytest tests
```

Note: with version 1.7.0 of the OpenDRIVE spec, the `Ex_LHT-Complex-X-Junction` file throws a connection position `ValueError` on plotting with this module. I think this is due to a mistake in the file; older versions of this file work without throwing an error, and I believe that the direction of the reference line of road `id==3` was reversed without swapping the corresponding `contactPoint`(s) - e.g. in road `id==34`, where it is listed as a successor with `contactPoint=="start"` despite the reference lines now traveling in opposite directions. If I'm incorrect about this and this is an error with the code, please raise an issue.

## TODO

Large components of the OpenDRIVE spec are not currently supported, including:
- Super Elevation / Road Shape (8.4.2 & 8.4.3)
- Road surfaces (8.5)
- Road markings (9.6)
- Junction groups (10.7)
- Objects (11)
- Signals (12)
- Railroads (13)
- `elementDir` / `elementS` for road & junction linking.

Some of these will be supported in the future. Pull requests implementing (or partially implementing) any of these are welcome. Additionally if you find anything else unsupported by this repo which is not covered by the above list (and there will be lots), please raise an issue.