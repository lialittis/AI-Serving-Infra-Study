const {chromium}=require('/tmp/inference-report-check/node_modules/playwright');
const fs=require('fs');
(async()=>{
 const browser=await chromium.launch({headless:true});
 const page=await browser.newPage({viewport:{width:1440,height:1100},offline:true});
 const errors=[];page.on('pageerror',e=>errors.push(String(e)));page.on('requestfailed',r=>errors.push(r.url()+': '+r.failure().errorText));
 const root='/home/tianchi/workspace/AI-Serving-Infra-Study/';
 await page.goto('file://'+root+'practice_19_kernel_core_usage/results/2026-09-28-run01/analysis/index.html');
 await page.locator('#tokens').selectOption('256');await page.locator('#mode').selectOption('pipe');await page.locator('#rep').selectOption('2');
 if(!(await page.locator('#task').textContent()).includes('Block Num 48'))throw Error('wrong task');
 if(!(await page.locator('#evidence').textContent()).includes('P19/pipe/tokens=256/rep=2'))throw Error('wrong association');
 await page.locator('summary').click();
 await page.screenshot({path:'/tmp/p19-report.png',fullPage:true});
 await page.setViewportSize({width:390,height:844});
 if(await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth))throw Error('mobile overflow');
 await page.locator('#tokens').selectOption('1');
 if(!(await page.locator('#task').textContent()).includes('Block Num 1 ·'))throw Error('wrong small task');
 let svgs=0;
 for(const mode of ['eager','graph'])for(const phase of ['prefill','decode-1','decode-2','decode-3']){
  await page.goto('file://'+root+`practice_15_kernel_execution_graph/results/2026-09-28-${mode}-run01/analysis/first_attention_${phase}.svg`);
  if(!(await page.locator('svg').textContent()).includes('AI_VECTOR_CORE / Block '+(phase==='prefill'?'10':'1')+' / Mix 0'))throw Error('missing P15 core field');
  svgs++;
 }
 if(errors.length)throw Error(errors.join('\n'));
 const result={offline:true,selectors_checked:['tokens','mode','rep'],mobile_overflow:false,p15_svgs:svgs,page_errors:errors};
 fs.writeFileSync('/tmp/p19-browser.json',JSON.stringify(result,null,2)+'\n');
 console.log(result);await browser.close();
})().catch(e=>{console.error(e);process.exit(1)});
