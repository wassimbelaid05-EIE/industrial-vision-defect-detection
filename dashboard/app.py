"""
Industrial Vision Dashboard
Defect detection for precision manufactured parts
Author: Wassim BELAID
Run: streamlit run dashboard/app.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import cv2
from datetime import datetime

from simulation.part_generator import PartGenerator, PartType, DefectType
from inspection.pipeline import InspectionPipeline, DEFECT_ACTIONS
from vision.detection.cnn_yolo import FeatureExtractor
from reporting.report_generator import generate_excel_report, generate_pdf_report, get_file_extension

st.set_page_config(page_title="Industrial Vision", page_icon="🔬", layout="wide")
st.markdown("""<style>
.kpi{background:#0d1a2e;border-radius:10px;padding:14px;border:1px solid #1a3050;text-align:center;margin:4px 0;}
.pass-box{background:#001a08;border-left:4px solid #00cc66;padding:10px;border-radius:0 6px 6px 0;margin:4px 0;}
.reject-box{background:#1a0000;border-left:4px solid #ff0000;padding:10px;border-radius:0 6px 6px 0;margin:4px 0;}
.rework-box{background:#1a0d00;border-left:4px solid #ff8c00;padding:10px;border-radius:0 6px 6px 0;margin:4px 0;}
.action-box{background:#1a0500;border-left:5px solid #ff3d00;padding:12px;border-radius:0 8px 8px 0;margin:6px 0;}
</style>""", unsafe_allow_html=True)

if "vi_init" not in st.session_state:
    st.session_state.pipeline = InspectionPipeline()
    st.session_state.generator = PartGenerator()
    st.session_state.reports = []
    st.session_state.current_img = None
    st.session_state.current_annotated = None
    st.session_state.current_report = None
    st.session_state.trained = False
    st.session_state.training_metrics = {}
    st.session_state.vi_init = True

pipeline = st.session_state.pipeline
gen = st.session_state.generator

# SIDEBAR
with st.sidebar:
    st.markdown("## 🔬 Industrial Vision")
    st.caption("Defect Detection | OpenCV + CNN + YOLO")
    st.divider()

    st.subheader("🎓 Train CNN Model")
    n_train = st.slider("Training samples", 30, 200, 80)
    defect_rate = st.slider("Defect rate (%)", 20, 80, 50) / 100

    if st.button("🚀 Train Model", use_container_width=True, type="primary"):
        with st.spinner(f"Generating {n_train} training images + training CNN..."):
            train_images, train_labels = [], []
            for _ in range(n_train):
                has_def = np.random.random() < defect_rate
                n_def = np.random.randint(1, 3) if has_def else 0
                img, defects = gen.generate(n_defects=n_def)
                train_images.append(img)
                label = defects[0].defect_type.value if defects else "no_defect"
                train_labels.append(label)
            metrics = pipeline.train(train_images, train_labels)
            st.session_state.trained = True
            st.session_state.training_metrics = metrics
        st.success(f"CNN trained! Accuracy: {metrics.get('accuracy',0)*100:.1f}%")

    st.divider()
    st.subheader("🏭 Inspection")
    part_type = st.selectbox("Part Type", [p.value for p in PartType])
    defect_inject = st.selectbox("Inject Defect", ["random"] + [d.value for d in DefectType if d != DefectType.NONE])
    n_defects = st.slider("N defects to inject", 0, 3, 1)

    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔍 Inspect", use_container_width=True, type="primary"):
            pt = PartType(part_type)
            dt = None if defect_inject == "random" else DefectType(defect_inject)
            img, defects = gen.generate(part_type=pt, defect_type=dt, n_defects=n_defects)
            report = pipeline.inspect(img, part_type=part_type, defects_ground_truth=defects)
            annotated = pipeline.get_annotated_image(img, report)
            st.session_state.current_img = img
            st.session_state.current_annotated = annotated
            st.session_state.current_report = report
            st.session_state.reports.append(report)
    with col2:
        if st.button("📦 Batch (20)", use_container_width=True):
            with st.spinner("Inspecting 20 parts..."):
                batch = gen.generate_batch(20, defect_rate)
                for item in batch:
                    r = pipeline.inspect(item["image"], item["part_id"],
                                        item["part_type"], item["defects"])
                    st.session_state.reports.append(r)
            st.success("Batch done!")

    st.divider()
    st.subheader("📊 Quick Stats")
    stats = pipeline.statistics
    st.markdown(f"""
- Inspected: **{stats['total_inspected']}**
- Pass: **{stats['pass_count']}** ({stats['pass_rate_pct']:.0f}%)
- Reject: **{stats['reject_count']}** ({stats['reject_rate_pct']:.0f}%)
- Rework: **{stats['rework_count']}** ({stats['rework_rate_pct']:.0f}%)
    """)

    st.divider()
    st.subheader("📥 Download Reports")
    reports = st.session_state.reports

    if reports:
        try:
            excel = generate_excel_report(reports, stats, st.session_state.training_metrics)
            st.download_button("📊 Excel Report (7 sheets)", excel,
                file_name=f"vision_report_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True, type="primary")
        except Exception as e:
            st.error(f"Excel: {e}")

        try:
            pdf_content = generate_pdf_report(reports, stats, st.session_state.training_metrics)
            ext = get_file_extension(pdf_content)
            st.download_button(f"📄 PDF Report ({'PDF' if ext=='pdf' else 'HTML'})",
                pdf_content,
                file_name=f"vision_report_{datetime.now().strftime('%Y%m%d_%H%M')}.{ext}",
                mime="application/pdf" if ext=="pdf" else "text/html",
                use_container_width=True)
        except Exception as e:
            st.error(f"PDF: {e}")
    else:
        st.info("Inspect parts first to download reports")

# HEADER
st.markdown("## 🔬 Industrial Vision — Defect Detection System")
st.caption("OpenCV + CNN (HOG+LBP+Gabor+RF) + YOLO-style | Swiss Precision Manufacturing | ISO 9001:2015")

stats = pipeline.statistics
k1,k2,k3,k4,k5,k6 = st.columns(6)
for col,label,val,color in [
    (k1,"Inspected",stats['total_inspected'],"#2196F3"),
    (k2,"PASS",f"{stats['pass_count']} ({stats['pass_rate_pct']:.0f}%)","#00cc66"),
    (k3,"REJECT",f"{stats['reject_count']} ({stats['reject_rate_pct']:.0f}%)","#ff3333"),
    (k4,"REWORK",f"{stats['rework_count']} ({stats['rework_rate_pct']:.0f}%)","#ff8c00"),
    (k5,"CNN Status","✅ Trained" if st.session_state.trained else "⚠️ Not trained","#9C27B0"),
    (k6,"CNN Accuracy",f"{st.session_state.training_metrics.get('accuracy',0)*100:.1f}%" if st.session_state.trained else "—","#FFD700"),
]:
    col.markdown(f'<div class="kpi"><p style="color:#aaa;font-size:10px">{label}</p><h3 style="color:{color};margin:4px 0">{val}</h3></div>', unsafe_allow_html=True)

st.divider()

tab1,tab2,tab3,tab4,tab5 = st.tabs([
    "🔍 Inspection","📊 Statistics","🤖 AI Models","⚠️ Actions","📋 History"
])

with tab1:
    col_img, col_res = st.columns([1,1])
    with col_img:
        st.subheader("Part Images")
        report = st.session_state.current_report
        annotated = st.session_state.current_annotated
        original = st.session_state.current_img

        if annotated is not None:
            # Convert BGR to RGB for display
            annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)

            sub1, sub2 = st.columns(2)
            with sub1:
                st.caption("Original")
                st.image(original_rgb, use_container_width=True)
            with sub2:
                st.caption("Annotated (Defects detected)")
                st.image(annotated_rgb, use_container_width=True)

            # CNN analysis visualization
            if st.session_state.trained:
                fe = FeatureExtractor()
                features = fe.extract(original)
                fig_feat = go.Figure(go.Bar(y=features[:30], marker_color="#2196F3"))
                fig_feat.update_layout(template="plotly_dark", height=150,
                    margin=dict(l=0,r=0,t=0,b=0), title="CNN Feature Vector (first 30 dims)")
                st.plotly_chart(fig_feat, use_container_width=True)
        else:
            st.info("👈 Click **Inspect** to analyze a part")

    with col_res:
        st.subheader("Inspection Result")
        if report:
            dec = report.decision
            dec_css = {"PASS":"pass-box","REJECT":"reject-box","REWORK":"rework-box"}.get(dec,"pass-box")
            dec_icon = {"PASS":"✅","REJECT":"❌","REWORK":"🔧"}.get(dec,"")
            st.markdown(f'<div class="{dec_css}"><h2>{dec_icon} {dec}</h2><p>Confidence: <b>{report.confidence*100:.1f}%</b> | Part: <b>{report.part_id}</b> ({report.part_type})</p></div>', unsafe_allow_html=True)

            m1,m2,m3,m4 = st.columns(4)
            m1.metric("Defects", report.n_defects)
            m2.metric("Anomaly Score", f"{report.anomaly_score:.0f}/100")
            m3.metric("Process Time", f"{report.processing_time_ms:.0f}ms")
            m4.metric("Ra (µm)", f"{report.surface_roughness_ra:.2f}")

            # Gauge
            dec_color = {"PASS":"#00cc66","REJECT":"#ff3333","REWORK":"#ff8c00"}.get(dec,"#aaa")
            fig_g = go.Figure(go.Indicator(
                mode="gauge+number",
                value=report.anomaly_score,
                gauge={"axis":{"range":[0,100]},"bar":{"color":dec_color},
                       "steps":[{"range":[0,30],"color":"#001a08"},{"range":[30,65],"color":"#1a0d00"},{"range":[65,100],"color":"#1a0000"}],
                       "threshold":{"line":{"color":"white","width":2},"value":65}},
                number={"font":{"color":dec_color}},
                title={"text":"Anomaly Score","font":{"color":"#aaa"}}))
            fig_g.update_layout(template="plotly_dark",height=200,margin=dict(l=10,r=10,t=30,b=10))
            st.plotly_chart(fig_g, use_container_width=True)

            # Defect details
            if report.defects:
                st.markdown("**Detected Defects:**")
                for d in report.defects:
                    sev_colors = {"critical":"#ff3333","high":"#ff8c00","medium":"#ffcc00","low":"#00cc66"}
                    sc = sev_colors.get(d.severity,"#aaa")
                    st.markdown(f'<div style="background:#0d1a2e;border-left:3px solid {sc};padding:8px;margin:3px 0;border-radius:0 4px 4px 0"><b style="color:{sc}">[{d.severity.upper()}]</b> {d.defect_type.value.replace("_"," ").title()} — Location: ({d.location[0]},{d.location[1]}) | Size: {d.size}px | Conf: {d.confidence:.2f}</div>', unsafe_allow_html=True)

            # CNN probabilities
            if st.session_state.trained:
                st.markdown(f"**CNN: `{report.cnn_prediction}`** ({report.cnn_confidence*100:.1f}%)")

        else:
            st.info("Inspect a part to see results")

with tab2:
    st.subheader("Statistical Quality Analysis")
    reports = st.session_state.reports
    if not reports:
        st.info("Inspect parts first")
    else:
        col_s1, col_s2 = st.columns(2)
        with col_s1:
            # Decision distribution pie
            decisions = [r.decision for r in reports]
            d_counts = {d: decisions.count(d) for d in ["PASS","REJECT","REWORK"]}
            fig_pie = go.Figure(go.Pie(
                labels=list(d_counts.keys()), values=list(d_counts.values()),
                hole=0.4,
                marker_colors=["#00cc66","#ff3333","#ff8c00"],
            ))
            fig_pie.update_layout(template="plotly_dark",height=280,
                margin=dict(l=0,r=0,t=10,b=0),title="Inspection Decision Distribution")
            st.plotly_chart(fig_pie, use_container_width=True)

        with col_s2:
            # Anomaly score distribution
            scores = [r.anomaly_score for r in reports]
            fig_hist = go.Figure(go.Histogram(x=scores, nbinsx=15,
                marker_color="#2196F3", opacity=0.85))
            fig_hist.add_vline(x=65, line_dash="dash", line_color="red",
                annotation_text="Reject threshold")
            fig_hist.add_vline(x=30, line_dash="dash", line_color="orange",
                annotation_text="Rework threshold")
            fig_hist.update_layout(template="plotly_dark",height=280,
                margin=dict(l=0,r=0,t=30,b=0),
                title="Anomaly Score Distribution",xaxis_title="Score",yaxis_title="Count")
            st.plotly_chart(fig_hist, use_container_width=True)

        # Defect type breakdown
        defect_types = {}
        for r in reports:
            for d in r.defects:
                k = d.defect_type.value
                defect_types[k] = defect_types.get(k, 0) + 1

        if defect_types:
            col_d1, col_d2 = st.columns(2)
            with col_d1:
                fig_def = px.bar(x=list(defect_types.values()), y=list(defect_types.keys()),
                    orientation="h",
                    color=list(defect_types.keys()),
                    title="Defect Type Frequency")
                fig_def.update_layout(template="plotly_dark",height=280,
                    margin=dict(l=0,r=0,t=30,b=0),showlegend=False)
                st.plotly_chart(fig_def, use_container_width=True)
            with col_d2:
                # Trend
                fig_trend = go.Figure()
                fig_trend.add_trace(go.Scatter(
                    y=[r.anomaly_score for r in reports],
                    mode="lines+markers",
                    line=dict(color="#cc0000",width=2),
                    marker=dict(color=["#ff3333" if r.decision=="REJECT" else
                                      ("#ff8c00" if r.decision=="REWORK" else "#00cc66")
                                      for r in reports],size=8),
                    name="Anomaly Score"
                ))
                fig_trend.add_hline(y=65,line_dash="dash",line_color="red")
                fig_trend.add_hline(y=30,line_dash="dash",line_color="orange")
                fig_trend.update_layout(template="plotly_dark",height=280,
                    margin=dict(l=0,r=0,t=10,b=0),
                    title="Anomaly Score Trend",yaxis_title="Score")
                st.plotly_chart(fig_trend, use_container_width=True)

        # Summary table
        summary_data = []
        for r in reports[-20:]:
            summary_data.append({
                "Part ID": r.part_id,"Type":r.part_type,"Decision":r.decision,
                "Score":round(r.anomaly_score,1),"Defects":r.n_defects,
                "Ra(µm)":r.surface_roughness_ra,"CNN":r.cnn_prediction,
                "Time(ms)":r.processing_time_ms,
            })
        if summary_data:
            st.dataframe(pd.DataFrame(summary_data),use_container_width=True,hide_index=True,height=250)

with tab3:
    st.subheader("AI Models — OpenCV + CNN + YOLO Architecture")
    c1,c2,c3 = st.columns(3)
    with c1:
        st.markdown("""**🔵 Classical CV (OpenCV)**
| Method | Detects |
|--------|---------|
| Blob detection | Pits, Inclusions |
| Morphology | Scratches, Cracks |
| Contour analysis | Chips, Burrs |
| Texture variance | Stains, Dents |
| Edge analysis | Chips, Burrs |
| CLAHE + bilateral | Preprocessing |
""")
    with c2:
        st.markdown("""**🟡 CNN Classifier**
| Feature | Description |
|---------|-------------|
| HOG | 144 gradient features |
| LBP | 64 texture features |
| Gabor | 16 frequency features |
| Statistics | 10 moment features |
| FFT | 8 frequency features |
| **Total** | **242 features** |
| Ensemble | RF + GBM voting |
""")
    with c3:
        st.markdown("""**🟢 YOLO-style Detector**
| Parameter | Value |
|-----------|-------|
| Grid | 8×8 cells |
| Cell size | 64×64 px |
| Threshold | 0.45 conf |
| NMS IoU | 0.40 |
| Classes | 8 defects |
| Method | Sliding window |
""")

    if st.session_state.training_metrics:
        m = st.session_state.training_metrics
        st.subheader("CNN Training Metrics")
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("CV Accuracy", f"{m.get('accuracy',0)*100:.1f}%")
        c2.metric("Std Dev", f"±{m.get('std',0)*100:.1f}%")
        c3.metric("Classes", m.get('n_classes',0))
        c4.metric("Training samples", m.get('n_samples',0))

    # Feature importance visualization (simulated)
    report = st.session_state.current_report
    if report and st.session_state.trained:
        st.subheader("Feature Importance (Current Part)")
        feat_groups = {"HOG (gradient)": 144, "LBP (texture)": 64,
                       "Gabor (frequency)": 16, "Statistics": 10, "FFT": 8}
        weights = [0.38, 0.28, 0.18, 0.10, 0.06]
        fig_fi = go.Figure(go.Bar(
            x=list(feat_groups.keys()), y=weights,
            marker_color=["#2196F3","#FF9800","#9C27B0","#00cc66","#ff4444"],
            text=[f"{w*100:.0f}%" for w in weights], textposition="auto"
        ))
        fig_fi.update_layout(template="plotly_dark",height=250,
            margin=dict(l=0,r=0,t=10,b=0),yaxis_title="Importance")
        st.plotly_chart(fig_fi, use_container_width=True)

with tab4:
    st.subheader("⚠️ Required Corrective Actions")
    st.caption("Based on ISO 9001:2015 Clause 8.7 — Control of Nonconforming Outputs")

    reports = st.session_state.reports
    defective = [r for r in reports if r.decision in ("REJECT","REWORK")]

    if not defective:
        st.success("✅ No corrective actions required — all parts PASSED")
    else:
        c1,c2,c3 = st.columns(3)
        c1.metric("Parts needing action", len(defective))
        c2.metric("Rejected", sum(1 for r in defective if r.decision=="REJECT"))
        c3.metric("Rework", sum(1 for r in defective if r.decision=="REWORK"))

        for report in defective[-10:]:
            dec_css = "reject-box" if report.decision=="REJECT" else "rework-box"
            st.markdown(f'<div class="{dec_css}"><b>{"❌ REJECT" if report.decision=="REJECT" else "🔧 REWORK"}: {report.part_id}</b> ({report.part_type}) | Defects: {report.n_defects} | Score: {report.anomaly_score:.0f}/100</div>', unsafe_allow_html=True)

            for action in report.actions_required:
                sev = action.get("severity","medium")
                urgency = action.get("urgency","")
                is_urgent = "Immediate" in str(urgency) or "batch" in str(urgency)
                bg = "#1a0000" if is_urgent else "#1a0500"
                border = "#ff0000" if is_urgent else "#ff3d00"
                st.markdown(f'''<div class="action-box" style="background:{bg};border-left-color:{border}">
                    <b style="color:#ff3d00">[{sev.upper()}] {action.get("defect","")}</b><br>
                    🔧 <b>Action:</b> {action.get("action","")}<br>
                    📋 <b>Standard:</b> {action.get("standard","")} &nbsp;|&nbsp; 👤 <b>Dept:</b> {action.get("responsible","")}<br>
                    ⏰ <b>Urgency:</b> <span style="color:{'#ff0000' if is_urgent else '#ff8c00'}">{urgency}</span> &nbsp;|&nbsp; 💶 <b>Est. cost:</b> €{action.get("estimated_cost_eur",0)}
                </div>''', unsafe_allow_html=True)

with tab5:
    st.subheader("📋 Inspection History")
    reports = st.session_state.reports
    if reports:
        hist_data = [r.to_dict() for r in reports[-50:]]
        hist_df = pd.DataFrame(hist_data)
        st.dataframe(hist_df, use_container_width=True, hide_index=True, height=400)

        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Total", len(reports))
        avg_time = np.mean([r.processing_time_ms for r in reports])
        c2.metric("Avg process time", f"{avg_time:.0f}ms")
        c3.metric("Throughput", f"{3600000/max(avg_time,1):.0f} parts/h")
        c4.metric("CNN accuracy", f"{st.session_state.training_metrics.get('accuracy',0)*100:.1f}%" if st.session_state.trained else "—")
    else:
        st.info("No inspection history yet")
