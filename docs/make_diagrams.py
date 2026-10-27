"""Writes docs/architecture.drawio (3 pages). Edit the boxes here, rerun, open the file in draw.io / app.diagrams.net.

    python3 docs/make_diagrams.py
"""
from pathlib import Path
from xml.sax.saxutils import escape

# colours per layer
C = {
    "source": ("#dae8fc", "#6c8ebf"), "ingest": ("#fff2cc", "#d6b656"), "store": ("#f5f5f5", "#666666"),
    "spark": ("#ffe6cc", "#d79b00"), "ml": ("#e1d5e7", "#9673a6"), "serve": ("#d5e8d4", "#82b366"),
    "devops": ("#f8cecc", "#b85450"), "note": ("#ffffff", "#999999"),
}


class Page:
    def __init__(self, name):
        self.name, self.cells, self.n = name, [], 2

    def _id(self):
        self.n += 1
        return f"c{self.n}"

    def group(self, label, x, y, w, h, kind):
        fill, line = C[kind]
        i = self._id()
        style = (f"rounded=1;whiteSpace=wrap;html=1;fillColor={fill};strokeColor={line};verticalAlign=top;"
                 "fontStyle=1;fontSize=14;arcSize=4;dashed=1;opacity=60;")
        self.cells.append(f'<mxCell id="{i}" value="{escape(label)}" style="{style}" vertex="1" parent="1">'
                          f'<mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" as="geometry"/></mxCell>')
        return i

    def box(self, label, x, y, w=170, h=60, kind="store", shape="rounded=1"):
        fill, line = C[kind]
        i = self._id()
        style = f"{shape};whiteSpace=wrap;html=1;fillColor={fill};strokeColor={line};fontSize=12;"
        self.cells.append(f'<mxCell id="{i}" value="{escape(label)}" style="{style}" vertex="1" parent="1">'
                          f'<mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" as="geometry"/></mxCell>')
        return i

    def db(self, label, x, y, w=150, h=70, kind="store"):
        return self.box(label, x, y, w, h, kind, "shape=cylinder3;boundedLbl=1;backgroundOutline=1;size=10")

    def text(self, label, x, y, w=300, h=40, size=12):
        i = self._id()
        style = f"text;html=1;whiteSpace=wrap;align=left;verticalAlign=top;fontSize={size};"
        self.cells.append(f'<mxCell id="{i}" value="{escape(label)}" style="{style}" vertex="1" parent="1">'
                          f'<mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" as="geometry"/></mxCell>')

    def arrow(self, a, b, label="", dashed=False, color="#333333", out=None, into=None):
        """out/into = (x, y) anchor on the source/target box, 0..1 from its top-left: keeps lines from crossing boxes"""
        i = self._id()
        style = (f"edgeStyle=orthogonalEdgeStyle;rounded=1;html=1;endArrow=block;strokeWidth=2;strokeColor={color};"
                 f"fontSize=11;labelBackgroundColor=#ffffff;{'dashed=1;' if dashed else ''}")
        if out:
            style += f"exitX={out[0]};exitY={out[1]};exitDx=0;exitDy=0;"
        if into:
            style += f"entryX={into[0]};entryY={into[1]};entryDx=0;entryDy=0;"
        self.cells.append(f'<mxCell id="{i}" value="{escape(label)}" style="{style}" edge="1" parent="1" '
                          f'source="{a}" target="{b}"><mxGeometry relative="1" as="geometry"/></mxCell>')

    def xml(self):
        body = "".join(self.cells)
        return (f'<diagram name="{escape(self.name)}"><mxGraphModel dx="1600" dy="900" grid="1" gridSize="10" '
                f'page="1" pageWidth="1700" pageHeight="1000"><root><mxCell id="0"/><mxCell id="1" parent="0"/>'
                f'{body}</root></mxGraphModel></diagram>')


# ---------------- page 1: the whole system ----------------
p = Page("1 · Whole system")
p.text("<b>Brussels transit: data platform, live status and ML</b><br>"
       "Arrows = data flow. Solid = running today · dashed = built, switched on with terraform apply", 20, 10, 900, 45, 16)

p.group("① Sources (open data)", 20, 70, 210, 440, "source")
stib = p.box("STIB API<br><i>waiting times + vehicles<br>every ~20 s</i>", 40, 110, kind="source")
air = p.box("IRCEL air quality<br><i>NO₂, PM, hourly</i>", 40, 200, kind="source")
elia = p.box("Elia grid load<br><i>since 2015, 15 min</i>", 40, 290, kind="source")
meteo = p.box("Open-Meteo weather<br><i>hourly</i>", 40, 380, kind="source")

p.group("② Ingestion", 260, 70, 230, 440, "ingest")
func = p.box("Azure Function<br><i>timer: STIB 5 min,<br>air + grid 15 min</i>", 290, 110, kind="ingest")
prod = p.box("Kafka producer<br><i>Container App, every 20 s</i><br>streaming/producer.py", 290, 230, h=70, kind="ingest")
kafka = p.box("Kafka = Azure Event Hubs<br>topics: stib-waiting,<br>stib-vehicles (1 day)", 290, 330, h=80, kind="ingest",
              shape="shape=process;size=0.08")

p.group("③ Storage + processing (Databricks / Spark)", 520, 70, 560, 440, "spark")
bronze = p.db("BRONZE<br>raw JSON in<br>Azure Data Lake", 545, 110, kind="store")
silver = p.db("SILVER<br>clean Delta tables", 730, 110, kind="store")
gold = p.db("GOLD<br>stop_line_hourly<br>city_hourly", 905, 110, w=160, kind="store")
batch = p.box("Batch job (daily 05:30)<br>notebooks/01_silver.sql<br>02_gold.sql · Spark SQL", 545, 225, w=200, h=65, kind="spark")
stream = p.box("Streaming job (every 15 min)<br>notebooks/03_stream_live_status.py<br>Spark Structured Streaming", 560, 330, w=230, h=70, kind="spark")
live = p.db("GOLD<br>line_status_live<br>normal / slower / disrupted", 880, 320, w=180, h=90, kind="store")
p.text("10-min windows per line,<br>compared with the usual<br>wait for that hour", 810, 420, 200, 50, 11)

p.group("④ Serving", 1110, 70, 260, 440, "serve")
js = p.box("line_status.json<br>lines.json<br><i>static website files</i>", 1140, 110, w=200, kind="serve")
api = p.box("FastAPI (Docker)<br>Container App, scales to 0<br>/network/status · /lines/{id}", 1140, 210, w=200, h=70, kind="serve")
web = p.box("lakehouse.jwds.me<br>live.jwds.me (map)", 1140, 310, w=200, kind="serve")
pbi = p.box("Power BI report", 1140, 410, w=200, h=50, kind="serve")

p.group("⑤ Machine learning + MLOps", 520, 540, 850, 200, "ml")
train = p.box("Weekly training job (Mondays)<br>Elia + weather → LightGBM,<br>neural net · tasks/grid_load", 545, 590, w=210, h=70, kind="ml")
mlf = p.box("MLflow tracking<br>params · metrics · plots · model", 795, 590, w=200, h=70, kind="ml")
reg = p.box("Model registry<br>brussels.ml.grid_load_lgbm<br>v1, v2 … @champion", 1035, 590, w=200, h=70, kind="ml")
gate = p.box("Gate: better than champion<br>on the same data?", 1035, 680, w=200, h=45, kind="ml", shape="rhombus")

p.group("⑥ DevOps (how code gets there)", 20, 540, 470, 200, "devops")
gh = p.box("GitHub repo<br>git push", 40, 590, w=130, kind="devops")
ci = p.box("GitHub Actions CI<br>pytest: helpers, producer,<br>API, promotion gate", 190, 590, w=150, h=70, kind="devops")
ghcr = p.box("ghcr.io<br>Docker images<br>→ API + producer", 360, 590, w=110, h=70, kind="devops")
tf = p.box("Terraform<br>infra/*.tf creates the Azure parts", 40, 680, w=430, h=40, kind="devops")

# running today (solid)
p.arrow(stib, func, "", out=(1, 0.3), into=(0, 0.25))
p.arrow(air, func, "", out=(1, 0.5), into=(0, 0.6))
p.arrow(elia, func, "", out=(1, 0.3), into=(0, 0.9))
p.arrow(func, bronze, "JSON every 5/15 min", into=(0, 0.3))
p.arrow(bronze, silver)
p.arrow(silver, gold)
p.arrow(gold, js, "lines.json", out=(1, 0.3), into=(0, 0.3))
p.arrow(js, web, "", out=(1, 0.5), into=(1, 0.5))
# built, switched on with terraform apply (dashed)
p.arrow(stib, prod, "every 20 s", dashed=True, out=(1, 0.85), into=(0, 0.3))
p.arrow(prod, kafka, "events", dashed=True)
p.arrow(kafka, stream, "read", dashed=True, into=(0, 0.5))
p.arrow(gold, stream, "usual wait", dashed=True, out=(0.4, 1), into=(0.9, 0))
p.arrow(stream, live, "", dashed=True)
p.arrow(live, js, "line_status.json", dashed=True, out=(1, 0.3), into=(0, 0.8))
p.arrow(js, api, "", dashed=True)
p.arrow(gold, pbi, "", dashed=True, out=(1, 0.85), into=(0, 0.5))
p.arrow(train, mlf, "log", dashed=True)
p.arrow(mlf, reg, "register", dashed=True)
p.arrow(reg, gate, "", dashed=True)
p.arrow(gh, ci, "", dashed=True)
p.arrow(ci, ghcr, "build", dashed=True)
page1 = p

# ---------------- page 2: MLOps loop ----------------
p = Page("2 · MLOps loop")
p.text("<b>MLOps: train → track → register → gate → serve → monitor → retrain</b><br>"
       "Analogy: a bakery. Recipe = code, batch = training run, lab notebook = MLflow, shelf = registry, "
       "cake on sale = @champion, new recipe being tasted = @challenger", 20, 10, 1300, 45, 15)
d = p.box("1 · Fresh data<br>get_data.py<br>(Elia + weather)", 40, 120, kind="source")
t = p.box("2 · Train<br>baselines, LightGBM,<br>neural network", 260, 120, kind="ml")
m = p.box("3 · Track (MLflow)<br>params · MAE/MAPE<br>plots · model file", 480, 120, kind="ml")
r = p.box("4 · Register<br>grid_load_lgbm vN", 700, 120, kind="ml")
g = p.box("5 · Gate<br>new vs @champion<br>on the SAME test hours", 920, 105, w=190, h=90, kind="ml", shape="rhombus")
ch = p.box("@champion<br>(live)", 1170, 60, w=130, h=55, kind="serve")
cl = p.box("@challenger<br>(kept, not live)", 1170, 170, w=130, h=55, kind="note")
s = p.box("6 · Serve<br>forecast job loads<br>models:/…@champion", 1150, 320, w=170, h=70, kind="serve")
mo = p.box("7 · Monitor (next)<br>daily forecast vs measured<br>error drifting?", 820, 320, w=200, h=70, kind="devops")
re = p.box("8 · Retrain<br>weekly job, Monday 06:00<br>or early on drift", 480, 320, w=200, h=70, kind="devops")
for a, b, lab in [(d, t, ""), (t, m, ""), (m, r, ""), (r, g, ""), (g, ch, "better"), (g, cl, "worse"),
                  (ch, s, ""), (s, mo, ""), (mo, re, "drift"), (re, d, "")]:
    p.arrow(a, b, lab)
p.text("<b>Where to see it</b><br>Databricks → Experiments (runs side by side) · Catalog → brussels → ml (versions, aliases)<br>"
       "Laptop demo: uv run mlflow ui --backend-store-uri sqlite:///outputs/mlflow.db → 127.0.0.1:5000<br>"
       "<b>Code</b>: src/mlkit.py (mlflow_setup, log_session, promote_if_better) · tests/test_promotion.py · "
       "infra/main.tf (job train-grid-load)", 40, 450, 1250, 80, 12)
page2 = p

# ---------------- page 3: CI/CD ----------------
p = Page("3 · CI/CD")
p.text("<b>CI/CD: from a commit on your laptop to a running container on Azure</b><br>"
       "Analogy: a factory exit. Every product is inspected (tests); only passing products are packed (Docker) and shipped (Azure)",
       20, 10, 1300, 45, 15)
a1 = p.box("You edit code<br>git commit", 40, 130, kind="devops")
a2 = p.box("git push<br>→ GitHub", 250, 130, kind="devops")
a3 = p.box("GitHub Actions: test<br>pytest tests/ + api/", 460, 130, w=180, kind="devops")
a4 = p.box("Tests fail ❌<br>nothing is shipped,<br>you get a red mark", 460, 270, w=180, h=70, kind="note")
a5 = p.box("On main: build<br>Docker images<br>(API, producer)", 690, 130, w=170, h=70, kind="devops")
a6 = p.box("ghcr.io<br>image registry<br>:latest + :commit", 910, 130, w=160, h=70, kind="devops")
a7 = p.box("Azure Container Apps<br>pull the image, run it<br>(defined in Terraform)", 1120, 130, w=190, h=70, kind="serve")
for a, b, lab in [(a1, a2, ""), (a2, a3, "every push"), (a3, a5, "pass ✅"), (a3, a4, "fail"), (a5, a6, "push"),
                  (a6, a7, "deploy")]:
    p.arrow(a, b, lab)
p.text("<b>Files</b>: .github/workflows/ci.yml · api/Dockerfile · streaming/Dockerfile · infra/streaming.tf<br>"
       "<b>See it</b>: github.com/Huntsama/brussels-transit-ml → Actions tab (every run, green or red)",
       40, 400, 1200, 50, 12)
page3 = p

out = Path(__file__).with_name("architecture.drawio")
out.write_text('<mxfile host="app.diagrams.net">' + page1.xml() + page2.xml() + page3.xml() + "</mxfile>")
print("wrote", out)
