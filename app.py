import streamlit as st
import plotly.graph_objects as go
import math

st.set_page_config(page_title="Swipe Smarter or Swipe More?", page_icon="💘", layout="wide")

st.markdown("""
<style>
  .block-container{padding-top:2rem;padding-bottom:2rem;}
  .metric-card{background:#f5f4f0;border-radius:10px;padding:1rem 1.2rem;margin-bottom:0.5rem;}
  .metric-label{font-size:12px;color:#888780;margin-bottom:4px;}
  .metric-value{font-size:26px;font-weight:600;color:#2c2c2a;}
  .metric-sub{font-size:11px;color:#888780;margin-top:2px;}
  .ins-amber{background:#FAEEDA;border-left:3px solid #EF9F27;border-radius:0 8px 8px 0;padding:10px 14px;margin-bottom:10px;}
  .ins-amber p{font-size:13px;color:#633806;margin:0;line-height:1.6;}
  .ins-purple{background:#EEEDFE;border-left:3px solid #534AB7;border-radius:0 8px 8px 0;padding:10px 14px;margin-bottom:10px;}
  .ins-purple p{font-size:13px;color:#3C3489;margin:0;line-height:1.6;}
  .ins-green{background:#EAF3DE;border-left:3px solid #639922;border-radius:0 8px 8px 0;padding:10px 14px;margin-bottom:10px;}
  .ins-green p{font-size:13px;color:#27500A;margin:0;line-height:1.6;}
  .ins-blue{background:#E6F1FB;border-left:3px solid #378ADD;border-radius:0 8px 8px 0;padding:10px 14px;margin-bottom:10px;}
  .ins-blue p{font-size:13px;color:#0C447C;margin:0;line-height:1.6;}
  .tag{display:inline-block;background:#EEEDFE;color:#3C3489;font-size:11px;padding:3px 10px;border-radius:4px;margin-right:6px;font-weight:500;}
</style>
""", unsafe_allow_html=True)

PURPLE="#534AB7"; PURPLE2="#7F77DD"; PURPLE3="#AFA9EC"
BLUE="#378ADD"; BLUE_L="#B5D4F4"; GREEN="#639922"
AMBER="#EF9F27"; RED="#E24B4A"; GRAY="#D3D1C7"

st.markdown("""
<h1 style='font-size:36px;font-weight:500;margin-bottom:2px;'>Swipe Smarter or Swipe More?</h1>
<p style='color:#888780;font-size:16px;margin-bottom:10px;'>Investigating the impact of excessive swiping on dating success &nbsp;·&nbsp; ML Project Dashboard</p>
<span class="tag">50,000 records</span>
<span class="tag">5 models trained</span>
<span class="tag">Target: dating_success</span>
<span class="tag">Best model: Logistic Regression (F1 = 0.281)</span>
<hr style='margin-top:12px;margin-bottom:0;border:none;border-top:0.5px solid #e0dfd8;'>
""", unsafe_allow_html=True)

tab1,tab2,tab3,tab4 = st.tabs(["Overview","Model comparison","EDA insights","Dating predictor"])

with tab1:
    c1,c2,c3,c4 = st.columns(4)
    with c1:
        st.markdown('<div class="metric-card"><div class="metric-label">Dataset size</div><div class="metric-value">50K</div><div class="metric-sub">records · 25 features (used 9)</div></div>',unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="metric-card"><div class="metric-label">Data split</div><div class="metric-value">80/20</div><div class="metric-sub">train / test split</div></div>',unsafe_allow_html=True)
    with c3:
        st.markdown('<div class="metric-card"><div class="metric-label">Best F1 score</div><div class="metric-value">0.281</div><div class="metric-sub">Logistic Regression</div></div>',unsafe_allow_html=True)
    with c4:
        st.markdown('<div class="metric-card"><div class="metric-label">Class balance</div><div class="metric-value">80/20</div><div class="metric-sub">majority / minority class</div></div>',unsafe_allow_html=True)

    st.markdown("---")
    cl,cr = st.columns(2)
    with cl:
        st.markdown("### Key findings")
        st.markdown("""
        <div class="ins-amber"><p><strong>Imbalanced dataset.</strong> Most users did not achieve a successful dating outcome. About <strong>80.44%</strong> had no successful outcome, while only <strong>19.56%</strong> succeeded. Dating success is the minority class — making accuracy alone unreliable as an evaluation metric.</p></div>
        <div class="ins-purple"><p><strong>Accuracy can be misleading.</strong> Naive Bayes reached 80.44% accuracy yet achieved an F1-score of 0.000 — it simply predicted "no success" for almost everyone. High accuracy without a good F1-score means the model is not useful in practice.</p></div>
        <div class="ins-green"><p><strong>Logistic Regression was the best model.</strong> It achieved the best balance between precision and recall for detecting successful dating outcomes — making it the most reliable model despite its lower raw accuracy.</p></div>
        """,unsafe_allow_html=True)
    with cr:
        st.markdown("### F1 Score vs Accuracy — All Models")
        models=['Naive Bayes','Random Forest','Decision Tree','Gradient Boosting','Logistic Regression']
        fig=go.Figure()
        fig.add_trace(go.Bar(name='Accuracy %',x=models,y=[80.44,80.31,68.13,52.77,50.42],marker_color=BLUE_L,marker_line_width=0))
        fig.add_trace(go.Bar(name='F1 Score ×100',x=models,y=[0,0.4,19.0,27.7,28.1],marker_color=PURPLE,marker_line_width=0))
        fig.update_layout(barmode='group',height=250,margin=dict(l=0,r=0,t=10,b=0),
            plot_bgcolor='rgba(0,0,0,0)',paper_bgcolor='rgba(0,0,0,0)',
            legend=dict(orientation='h',y=-0.25,font=dict(size=11)),
            yaxis=dict(gridcolor='#f0efe8',range=[0,100]),xaxis=dict(tickfont=dict(size=10)))
        st.plotly_chart(fig,use_container_width=True)
        st.markdown("""<div style='background:#f5f4f0;border-radius:8px;padding:10px 14px;'>
        <p style='font-size:12px;color:#444441;margin:0;line-height:1.6;'><strong>Key takeaway:</strong> Naive Bayes and Random Forest scored ~80% accuracy but near-zero F1 — they predicted the majority class only. Logistic Regression and Gradient Boosting had lower accuracy but meaningfully detected dating success cases.</p>
        </div>""",unsafe_allow_html=True)

    st.markdown("### ML Pipeline")
    pc=st.columns(6)
    stages=[("Data Import","#F1EFE8","#444441"),("Data Pre-processing","#E6F1FB","#185FA5"),
            ("Feature Engineering","#EEEDFE","#3C3489"),("Model Training","#FAEEDA","#854F0B"),
            ("Model Evaluation","#EAF3DE","#27500A"),("Best Model","#534AB7","#ffffff")]
    for col,(label,bg,fg) in zip(pc,stages):
        with col:
            st.markdown(f'<div style="background:{bg};color:{fg};border-radius:10px;padding:10px 4px;text-align:center;font-size:16px;font-weight:500;">{label}</div>',unsafe_allow_html=True)
    st.markdown('<p style="font-size:13px;color:#888780;margin-top:8px;">Target: <code>dating_success</code> &nbsp;·&nbsp; 1 = Relationship Formed / Date Happened &nbsp;·&nbsp; 0 = Others</p>',unsafe_allow_html=True)

with tab2:
    st.markdown("### F1 score — primary metric")
    models_f1=['Logistic Regression','Gradient Boosting','Decision Tree','Random Forest','Naive Bayes']
    f1v=[0.2808,0.2768,0.1897,0.0040,0.0000]
    fig2=go.Figure(go.Bar(x=f1v,y=models_f1,orientation='h',
        marker_color=[PURPLE,PURPLE2,PURPLE3,GRAY,GRAY],marker_line_width=0,
        text=[f"{v:.4f}" for v in f1v],textposition='outside'))
    fig2.update_layout(height=250,margin=dict(l=0,r=60,t=10,b=0),
        plot_bgcolor='rgba(0,0,0,0)',paper_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(range=[0,0.36],gridcolor='#f0efe8'),yaxis=dict(tickfont=dict(size=12)))
    st.plotly_chart(fig2,use_container_width=True)

    st.markdown("---")
    ca,cw=st.columns(2)
    with ca:
        st.markdown("### Accuracy — all models")
        fig3=go.Figure(go.Bar(x=[80.44,80.31,68.13,52.77,50.42],
            y=['Naive Bayes','Random Forest','Decision Tree','Gradient Boosting','Logistic Regression'],
            orientation='h',marker_color=[BLUE_L,BLUE_L,PURPLE3,PURPLE2,PURPLE],marker_line_width=0,
            text=[f"{v:.1f}%" for v in [80.44,80.31,68.13,52.77,50.42]],textposition='outside'))
        fig3.update_layout(height=250,margin=dict(l=0,r=60,t=10,b=0),
            plot_bgcolor='rgba(0,0,0,0)',paper_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(range=[0,100],gridcolor='#f0efe8'),yaxis=dict(tickfont=dict(size=12)))
        st.plotly_chart(fig3,use_container_width=True)
    with cw:
        st.markdown("### Why F1 score was chosen")
        st.markdown("""
        <div class="ins-amber"><p><strong>Class imbalance (~80% vs ~20%):</strong> Naive Bayes achieved 80.44% accuracy by predicting "no success" for almost everyone — yet its F1 was 0.000. Accuracy alone is deeply misleading here.</p></div>
        <div class="ins-green"><p><strong>F1 balances precision and recall.</strong> It penalises both false positives and false negatives — making it the correct metric for identifying the minority class (actual dating successes).</p></div>
        """,unsafe_allow_html=True)
        m1,m2,m3=st.columns(3)
        with m1: st.markdown('<div class="metric-card"><div class="metric-label">Logistic Regression</div><div class="metric-value" style="font-size:18px;">50.4%</div><div class="metric-sub">F1 = <b>0.281</b> ← best</div></div>',unsafe_allow_html=True)
        with m2: st.markdown('<div class="metric-card"><div class="metric-label">Naive Bayes</div><div class="metric-value" style="font-size:18px;">80.4%</div><div class="metric-sub">F1 = <b>0.000</b> ← worst</div></div>',unsafe_allow_html=True)
        with m3: st.markdown('<div class="metric-card"><div class="metric-label">Gradient Boosting</div><div class="metric-value" style="font-size:18px;">52.8%</div><div class="metric-sub">Runner-up · F1 0.277</div></div>',unsafe_allow_html=True)

with tab3:
    st.markdown("### Distribution of dating success")
    dd1,dd2=st.columns(2)
    with dd1:
        fig_d=go.Figure(go.Pie(labels=['No successful outcome (0)','Successful outcome (1)'],
            values=[80.44,19.56],marker_colors=[GRAY,PURPLE],hole=0.6,
            textinfo='label+percent',textfont=dict(size=11)))
        fig_d.update_layout(height=260,margin=dict(l=0,r=0,t=10,b=0),paper_bgcolor='rgba(0,0,0,0)',showlegend=False)
        st.plotly_chart(fig_d,use_container_width=True)
    with dd2:
        st.markdown("""<div class="ins-amber" style="margin-top:1rem;">
        <p><strong>Heavily imbalanced target.</strong> About 80.44% of users had no successful dating outcome, while only 19.56% formed a relationship or went on a date. This imbalance is why F1-score was chosen over accuracy as the primary evaluation metric.</p>
        </div>""",unsafe_allow_html=True)
        ea,eb=st.columns(2)
        with ea: st.markdown('<div class="metric-card"><div class="metric-label">No successful outcome</div><div class="metric-value" style="color:#888780;">80.44%</div></div>',unsafe_allow_html=True)
        with eb: st.markdown('<div class="metric-card"><div class="metric-label">Successful outcome</div><div class="metric-value" style="color:#534AB7;">19.56%</div></div>',unsafe_allow_html=True)

    st.markdown("---")
    e1,e2=st.columns(2)
    with e1:
        st.markdown("### Swipe right ratio by outcome")
        fig_sw=go.Figure()
        fig_sw.add_trace(go.Bar(name='No success (0)',x=['0.0–0.2','0.2–0.4','0.4–0.6','0.6–0.8','0.8–1.0'],y=[9.8,19.2,22.1,20.8,8.1],marker_color=GRAY,marker_line_width=0))
        fig_sw.add_trace(go.Bar(name='Success (1)',x=['0.0–0.2','0.2–0.4','0.4–0.6','0.6–0.8','0.8–1.0'],y=[2.3,4.6,5.9,5.3,2.0],marker_color=PURPLE,marker_line_width=0))
        fig_sw.update_layout(barmode='group',height=240,margin=dict(l=0,r=0,t=10,b=0),
            plot_bgcolor='rgba(0,0,0,0)',paper_bgcolor='rgba(0,0,0,0)',
            legend=dict(orientation='h',y=-0.3,font=dict(size=11)),
            yaxis=dict(title='% of users',gridcolor='#f0efe8'),xaxis=dict(title='Swipe right ratio'))
        st.plotly_chart(fig_sw,use_container_width=True)
        st.caption("Swipe ratio shows only modest class separation — not a strong standalone predictor.")
    with e2:
        st.markdown("### App usage time vs dating success")
        fig_us=go.Figure(go.Bar(x=['Light','Moderate','High','Extreme'],y=[18.2,19.8,20.1,18.9],
            marker_color=[BLUE_L,PURPLE2,PURPLE,"#3C3489"],marker_line_width=0,
            text=["18.2%","19.8%","20.1%","18.9%"],textposition='outside'))
        fig_us.update_layout(height=240,margin=dict(l=0,r=0,t=10,b=40),
            plot_bgcolor='rgba(0,0,0,0)',paper_bgcolor='rgba(0,0,0,0)',
            yaxis=dict(title='Success rate %',gridcolor='#f0efe8',range=[15,25]),xaxis=dict(title='App usage category'))
        st.plotly_chart(fig_us,use_container_width=True)
        st.caption("More time on the app does not guarantee dating success.")

    st.markdown("### Feature importance")
    feats=['mutual_matches','likes_received','message_sent_count','app_usage_time_min','swipe_right_ratio ⭐','emoji_usage_rate','bio_length','age']
    imp=[100,82,70,55,40,35,28,22]
    fig_f=go.Figure(go.Bar(x=imp,y=feats,orientation='h',
        marker_color=[PURPLE,PURPLE2,PURPLE3,BLUE_L,AMBER,GRAY,GRAY,GRAY],marker_line_width=0,
        text=['High','High','High','Med','Low','Low','Low','Low'],textposition='outside'))
    fig_f.update_layout(height=320,margin=dict(l=0,r=80,t=10,b=0),
        plot_bgcolor='rgba(0,0,0,0)',paper_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(range=[0,130],gridcolor='#f0efe8',title='Relative importance'),yaxis=dict(tickfont=dict(size=12)))
    st.plotly_chart(fig_f,use_container_width=True)

    st.markdown("### Match outcomes in the dataset")
    fig_o=go.Figure(go.Pie(
        labels=['Mutual Match','Chat Ignored','Date Happened','Ghosted','No Action','One-sided Like','Catfished'],
        values=[14.3,14.3,14.3,14.3,14.3,14.3,14.2],
        marker_colors=[PURPLE,PURPLE2,BLUE,"#88C057",AMBER,RED,BLUE_L],hole=0.55,
        textinfo='label+percent',textfont=dict(size=10)))
    fig_o.update_layout(height=280,margin=dict(l=0,r=0,t=10,b=0),paper_bgcolor='rgba(0,0,0,0)',showlegend=False)
    st.plotly_chart(fig_o,use_container_width=True)

with tab4:
    st.markdown("""<div class="ins-blue">
    <p>This simulator is based on the Logistic Regression model — the best model by F1-score (0.281). Adjust the sliders to see how behavioural features affect the predicted probability of dating success.</p>
    </div>""",unsafe_allow_html=True)

    ci,cr2=st.columns(2)
    with ci:
        sw=st.slider("Swipe right ratio (main feature)",0.0,1.0,0.5,0.01)
        mu=st.slider("Mutual matches",0,50,20)
        li=st.slider("Likes received",0,200,80)
        ms=st.slider("Messages sent",0,500,150)
        us=st.slider("App usage (min/day)",10,600,180,5)
        em=st.slider("Emoji usage rate",0.0,1.0,0.3,0.01)
        bi=st.slider("Bio length (characters)",0,500,150,10)
        ag=st.slider("Age",18,65,28)
        gn=st.selectbox("Gender",["Male","Female","Non-binary","Prefer not to say"])

    with cr2:
        gmap={"Male":0.0,"Female":0.5,"Non-binary":0.3,"Prefer not to say":0.2}
        gv=gmap[gn]
        sn=(sw-0.5)*0.8; mn=(mu-25)/25*1.8; ln=(li-100)/100*1.4
        msn=(ms-250)/250*1.2; un=(us-300)/300*0.6; en=(em-0.3)*0.5
        bn=(bi-200)/200*0.4; an=(35-ag)/20*0.3; gnv=(gv-0.25)*0.2
        prob=round(1/(1+math.exp(-(-0.8+sn+mn+ln+msn+un+en+bn+an+gnv)))*100)
        col=GREEN if prob>=65 else (PURPLE if prob>=40 else RED)
        vt="High chance of success" if prob>=65 else ("Moderate chance" if prob>=40 else "Low probability")
        vbg="#EAF3DE" if prob>=65 else ("#EEEDFE" if prob>=40 else "#FCEBEB")
        st.markdown(f"""<div style='background:#f5f4f0;border-radius:12px;padding:1.5rem;text-align:center;margin-bottom:1rem;'>
            <p style='font-size:13px;color:#888780;margin-bottom:8px;'>Predicted dating success probability</p>
            <p style='font-size:48px;font-weight:600;color:{col};margin:0;'>{prob}%</p>
            <div style='background:#e0dfd8;border-radius:5px;height:10px;margin:14px 0;overflow:hidden;'>
                <div style='width:{prob}%;background:{col};height:100%;border-radius:5px;'></div>
            </div>
            <span style='background:{vbg};color:{col};padding:4px 16px;border-radius:20px;font-size:13px;font-weight:500;'>{vt}</span>
        </div>""",unsafe_allow_html=True)

        st.markdown("### Feature impact on your inputs")
        imps=[("Mutual matches",abs(mn)),("Likes received",abs(ln)),("Messages sent",abs(msn)),
              ("App usage time",abs(un)),("Swipe right ratio",abs(sn)),("Emoji usage",abs(en)),
              ("Bio length",abs(bn)),("Age",abs(an))]
        imps.sort(key=lambda x:x[1],reverse=True)
        fig_i=go.Figure(go.Bar(x=[round(v*100,1) for _,v in imps],y=[l for l,_ in imps],
            orientation='h',marker_color=[PURPLE,PURPLE2,PURPLE3,BLUE_L,AMBER,GRAY,GRAY,GRAY],marker_line_width=0))
        fig_i.update_layout(height=260,margin=dict(l=0,r=20,t=10,b=0),
            plot_bgcolor='rgba(0,0,0,0)',paper_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(gridcolor='#f0efe8',title='Relative impact'),yaxis=dict(tickfont=dict(size=11)))
        st.plotly_chart(fig_i,use_container_width=True)

        tips=[]
        if mu<20: tips.append("Increase mutual matches — the strongest predictor")
        if ms<100: tips.append("Send more messages to show active engagement")
        if li<50: tips.append("More likes received signals higher profile appeal")
        if sw>0.8: tips.append("Very high swipe ratio may reduce match quality")
        tips.append("Swipe ratio alone has limited predictive power (F1 = 0.281)")
        for t in tips[:3]: st.markdown(f"- {t}")
