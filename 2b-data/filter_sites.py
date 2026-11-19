"""Stream each VicRoads Traffic Signal Volume zip and keep only the 40 Boroondara
SCATS sites used in the COS30019 2B October 2006 dataset. Writes one csv.gz per zip."""
import io, subprocess, sys, tempfile, zipfile, pathlib
import pandas as pd

SITES = {970, 2000, 2200, 2820, 2825, 2827, 2846, 3001, 3002, 3120, 3122, 3126, 3127,
         3180, 3662, 3682, 3685, 3804, 3812, 4030, 4032, 4034, 4035, 4040, 4043, 4051,
         4057, 4063, 4262, 4263, 4264, 4266, 4270, 4272, 4273, 4321, 4324, 4335, 4812, 4821}
COLUMNS = (["NB_SCATS_SITE", "QT_INTERVAL_COUNT", "NB_DETECTOR"]
           + [f"V{i:02d}" for i in range(96)]
           + ["NM_REGION", "CT_RECORDS", "QT_VOLUME_24HOUR", "CT_ALARM_24HOUR"])

def open_member(path, z, name):
    try:
        return z.open(name)
    except NotImplementedError:
        # 2019 archive uses Deflate64, which zipfile can't read; the system unzip can
        return io.BytesIO(subprocess.run(["unzip", "-p", str(path), name],
                                         check=True, capture_output=True).stdout)

def read_csvs(path, parts):
    # Some yearly archives (2018, 2020-2024) contain monthly zips instead of CSVs
    with zipfile.ZipFile(path) as z:
        for name in sorted(z.namelist()):
            if name.lower().endswith(".csv"):
                df = pd.read_csv(open_member(path, z, name), low_memory=False)
                # 2015 includes a few files in another format (no VNN columns); skip them
                if "V00" in df.columns:
                    parts.append(df.loc[df.NB_SCATS_SITE.isin(SITES), COLUMNS])
            elif name.lower().endswith(".zip"):
                with tempfile.NamedTemporaryFile(suffix=".zip") as tmp:
                    tmp.write(open_member(path, z, name).read())
                    tmp.flush()
                    read_csvs(tmp.name, parts)

def filter_zip(src, out_dir):
    out = out_dir / (src.stem + "_boroondara.csv.gz")
    if out.exists():
        return out
    parts = []
    read_csvs(src, parts)
    pd.concat(parts).to_csv(out.with_suffix(".tmp"), index=False, compression="gzip")
    out.with_suffix(".tmp").rename(out)
    return out

if __name__ == "__main__":
    out_dir = pathlib.Path(sys.argv[2])
    for src in sorted(pathlib.Path(sys.argv[1]).glob("*.zip")):
        print(filter_zip(src, out_dir), flush=True)
