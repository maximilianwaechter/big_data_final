# Heat Vulnerability × Crime in Philadelphia

Final project for Big Data, Spring 2026.
Bivariate spatial analysis combining Landsat 8 land-surface temperature with
2024 Philadelphia Police Department incident records on a 500 m hex grid.

## Repository contents

| Path | Purpose |
|---|---|
| `heat_crime_final_project.ipynb` | Final-report notebook (executive summary + 5 project areas + figures + references) |
| `heat_crime_final_project.pdf`   | PDF export of the executed notebook |
| `data/`                          | Processed GeoJSON inputs (hexes, crime points, stations, metadata) |
| `figures/`                       | Static figures exported by the notebook |
| `web/index.html`                 | Interactive Leaflet map (deployed via GitHub Pages) |
| `environment.yml`                | Conda environment spec to reproduce the notebook |

## Live interactive map

👉 **<https://maximilianwaechter.github.io/big_data_final/>**  

Two tabs: a bivariate **Heat × Crime** view (red = hot, blue = high-crime, purple = both)
and a side-by-side **Comparison** view. Click any of the 20 PPD station markers for
captain, division, address, phone, and email.

## Reproducing the notebook

```bash
conda env create -f environment.yml
conda activate philly-heat-crime
jupyter lab heat_crime_final_project.ipynb
```

Then **Run all** to regenerate every figure.

## Data sources

* **Landsat 8 Collection 2 Level-2, band ST_B10** — Microsoft Planetary Computer
  (scene `LC08_L2SP_014032_20240903_02_T1`)
* **2024 PPD incidents** — OpenDataPhilly (`phl.carto.com`)
* **PPD district stations** — official phillypolice.com pages, geocoded with OpenStreetMap Nominatim
* **City boundary** — OpenDataPhilly
