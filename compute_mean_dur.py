<!DOCTYPE html><html lang="en"><head><meta charset="utf-8"><meta http-equiv="X-UA-Compatible" content="IE=edge"><title>Source of compute_mean_dur.py - UVAST_final - Social Coding</title><script>
window.WRM=window.WRM||{};window.WRM._unparsedData=window.WRM._unparsedData||{};window.WRM._unparsedErrors=window.WRM._unparsedErrors||{};
WRM._unparsedData["com.atlassian.plugins.atlassian-plugins-webresource-plugin:context-path.context-path"]="\u0022\u0022";
WRM._unparsedData["com.atlassian.bitbucket.server.bitbucket-webpack-INTERNAL:date-format-preference.data"]="\u0022RELATIVE\u0022";
WRM._unparsedData["com.atlassian.bitbucket.server.bitbucket-webpack-INTERNAL:determine-language.syntax-highlighters"]="{\u0022text/x-ruby\u0022:{\u0022x\u0022:[\u0022ruby\u0022]},\u0022text/x-c++src\u0022:{\u0022e\u0022:[\u0022inl\u0022]},\u0022text/x-objectivec\u0022:{\u0022e\u0022:[\u0022m\u0022]},\u0022text/x-python\u0022:{\u0022x\u0022:[\u0022python\u0022]},\u0022text/javascript\u0022:{\u0022x\u0022:[\u0022node\u0022]},\u0022text/x-sh\u0022:{\u0022e\u0022:[\u0022makefile\u0022,\u0022Makefile\u0022],\u0022x\u0022:[\u0022sh\u0022,\u0022bash\u0022,\u0022zsh\u0022]},\u0022text/x-perl\u0022:{\u0022x\u0022:[\u0022perl\u0022]},\u0022text/velocity\u0022:{\u0022e\u0022:[\u0022vm\u0022]},\u0022text/x-erlang\u0022:{\u0022x\u0022:[\u0022escript\u0022]}}";
WRM._unparsedData["com.atlassian.bitbucket.server.feature-wrm-data:bidi.character.highlighting.data"]="true";
WRM._unparsedData["com.atlassian.bitbucket.server.bitbucket-webpack-INTERNAL:user-keyboard-shortcuts-enabled.data"]="true";
WRM._unparsedData["com.atlassian.bitbucket.server.config-wrm-data:attachment.upload.max.size.data"]="{\u0022value\u0022:\u002210485760\u0022,\u0022key\u0022:\u0022attachment.upload.max.size\u0022,\u0022type\u0022:\u0022NUMBER\u0022}";
WRM._unparsedData["com.atlassian.bitbucket.server.feature-wrm-data:attachments.data"]="true";
WRM._unparsedData["com.atlassian.applinks.applinks-plugin:applinks-common-exported.applinks-help-paths"]="{\u0022entries\u0022:{\u0022applinks.docs.root\u0022:\u0022https://confluence.atlassian.com/display/APPLINKS-072/\u0022,\u0022applinks.docs.diagnostics.troubleshoot.sslunmatched\u0022:\u0022SSL+and+application+link+troubleshooting+guide\u0022,\u0022applinks.docs.diagnostics.troubleshoot.oauthsignatureinvalid\u0022:\u0022OAuth+troubleshooting+guide\u0022,\u0022applinks.docs.diagnostics.troubleshoot.oauthtimestamprefused\u0022:\u0022OAuth+troubleshooting+guide\u0022,\u0022applinks.docs.delete.entity.link\u0022:\u0022Create+links+between+projects\u0022,\u0022applinks.docs.adding.application.link\u0022:\u0022Link+Atlassian+applications+to+work+together\u0022,\u0022applinks.docs.administration.guide\u0022:\u0022Application+Links+Documentation\u0022,\u0022applinks.docs.oauth.security\u0022:\u0022OAuth+security+for+application+links\u0022,\u0022applinks.docs.troubleshoot.application.links\u0022:\u0022Troubleshoot+application+links\u0022,\u0022applinks.docs.diagnostics.troubleshoot.unknownerror\u0022:\u0022Network+and+connectivity+troubleshooting+guide\u0022,\u0022applinks.docs.configuring.auth.trusted.apps\u0022:\u0022Configuring+Trusted+Applications+authentication+for+an+application+link\u0022,\u0022applinks.docs.diagnostics.troubleshoot.authlevelunsupported\u0022:\u0022OAuth+troubleshooting+guide\u0022,\u0022applinks.docs.diagnostics.troubleshoot.ssluntrusted\u0022:\u0022SSL+and+application+link+troubleshooting+guide\u0022,\u0022applinks.docs.diagnostics.troubleshoot.unknownhost\u0022:\u0022Network+and+connectivity+troubleshooting+guide\u0022,\u0022applinks.docs.delete.application.link\u0022:\u0022Link+Atlassian+applications+to+work+together\u0022,\u0022applinks.docs.adding.project.link\u0022:\u0022Configuring+Project+links+across+Applications\u0022,\u0022applinks.docs.link.applications\u0022:\u0022Link+Atlassian+applications+to+work+together\u0022,\u0022applinks.docs.diagnostics.troubleshoot.oauthproblem\u0022:\u0022OAuth+troubleshooting+guide\u0022,\u0022applinks.docs.diagnostics.troubleshoot.migration\u0022:\u0022Update+application+links+to+use+OAuth\u0022,\u0022applinks.docs.relocate.application.link\u0022:\u0022Link+Atlassian+applications+to+work+together\u0022,\u0022applinks.docs.administering.entity.links\u0022:\u0022Create+links+between+projects\u0022,\u0022applinks.docs.upgrade.application.link\u0022:\u0022OAuth+security+for+application+links\u0022,\u0022applinks.docs.diagnostics.troubleshoot.connectionrefused\u0022:\u0022Network+and+connectivity+troubleshooting+guide\u0022,\u0022applinks.docs.configuring.auth.oauth\u0022:\u0022OAuth+security+for+application+links\u0022,\u0022applinks.docs.insufficient.remote.permission\u0022:\u0022OAuth+security+for+application+links\u0022,\u0022applinks.docs.configuring.application.link.auth\u0022:\u0022OAuth+security+for+application+links\u0022,\u0022applinks.docs.diagnostics\u0022:\u0022Application+links+diagnostics\u0022,\u0022applinks.docs.configured.authentication.types\u0022:\u0022OAuth+security+for+application+links\u0022,\u0022applinks.docs.adding.entity.link\u0022:\u0022Create+links+between+projects\u0022,\u0022applinks.docs.diagnostics.troubleshoot.unexpectedresponse\u0022:\u0022Network+and+connectivity+troubleshooting+guide\u0022,\u0022applinks.docs.configuring.auth.basic\u0022:\u0022Configuring+Basic+HTTP+Authentication+for+an+Application+Link\u0022,\u0022applinks.docs.diagnostics.troubleshoot.authlevelmismatch\u0022:\u0022OAuth+troubleshooting+guide\u0022}}";
WRM._unparsedData["com.atlassian.applinks.applinks-plugin:applinks-common-exported.applinks-types"]="{\u0022crowd\u0022:\u0022Crowd\u0022,\u0022confluence\u0022:\u0022Confluence\u0022,\u0022remote.plugin.container\u0022:\u0022Atlassian Connect\u0022,\u0022fecru\u0022:\u0022FishEye / Crucible\u0022,\u0022stash\u0022:\u0022Bitbucket Server\u0022,\u0022jira\u0022:\u0022Jira\u0022,\u0022refapp\u0022:\u0022Reference Application\u0022,\u0022bamboo\u0022:\u0022Bamboo\u0022,\u0022generic\u0022:\u0022Generic Application\u0022}";
WRM._unparsedData["com.atlassian.applinks.applinks-plugin:applinks-common-exported.entity-types"]="{\u0022singular\u0022:{\u0022refapp.charlie\u0022:\u0022Charlie\u0022,\u0022fecru.project\u0022:\u0022Crucible Project\u0022,\u0022fecru.repository\u0022:\u0022FishEye Repository\u0022,\u0022stash.project\u0022:\u0022Bitbucket Server Project\u0022,\u0022generic.entity\u0022:\u0022Generic Project\u0022,\u0022confluence.space\u0022:\u0022Confluence Space\u0022,\u0022bamboo.project\u0022:\u0022Bamboo Project\u0022,\u0022jira.project\u0022:\u0022Jira Project\u0022},\u0022plural\u0022:{\u0022refapp.charlie\u0022:\u0022Charlies\u0022,\u0022fecru.project\u0022:\u0022Crucible Projects\u0022,\u0022fecru.repository\u0022:\u0022FishEye Repositories\u0022,\u0022stash.project\u0022:\u0022Bitbucket Server Projects\u0022,\u0022generic.entity\u0022:\u0022Generic Projects\u0022,\u0022confluence.space\u0022:\u0022Confluence Spaces\u0022,\u0022bamboo.project\u0022:\u0022Bamboo Projects\u0022,\u0022jira.project\u0022:\u0022Jira Projects\u0022}}";
WRM._unparsedData["com.atlassian.applinks.applinks-plugin:applinks-common-exported.authentication-types"]="{\u0022com.atlassian.applinks.api.auth.types.BasicAuthenticationProvider\u0022:\u0022Basic Access\u0022,\u0022com.atlassian.applinks.api.auth.types.TrustedAppsAuthenticationProvider\u0022:\u0022Trusted Applications\u0022,\u0022com.atlassian.applinks.api.auth.types.CorsAuthenticationProvider\u0022:\u0022CORS\u0022,\u0022com.atlassian.applinks.api.auth.types.OAuthAuthenticationProvider\u0022:\u0022OAuth\u0022,\u0022com.atlassian.applinks.api.auth.types.TwoLeggedOAuthAuthenticationProvider\u0022:\u0022OAuth\u0022,\u0022com.atlassian.applinks.api.auth.types.TwoLeggedOAuthWithImpersonationAuthenticationProvider\u0022:\u0022OAuth\u0022}";
WRM._unparsedData["com.atlassian.bitbucket.server.bitbucket-client-web-fragments:comments-action-links._unused_"]="null";
WRM._unparsedData["com.atlassian.bitbucket.server.bitbucket-client-web-fragments:comments-info-panels._unused_"]="null";
WRM._unparsedData["com.atlassian.bitbucket.server.bitbucket-client-web-fragments:comments-extra-panels-internal._unused_"]="null";
WRM._unparsedData["com.atlassian.bitbucket.server.bitbucket-client-web-fragments:file-content-diff-view-options._unused_"]="null";
WRM._unparsedData["com.atlassian.bitbucket.server.config-wrm-data:display.max.source.lines.data"]="{\u0022value\u0022:\u002220000\u0022,\u0022key\u0022:\u0022display.max.source.lines\u0022,\u0022type\u0022:\u0022NUMBER\u0022}";
WRM._unparsedData["com.atlassian.bitbucket.server.config-wrm-data:feature.pull.request.suggestions.data"]="{\u0022value\u0022:\u0022true\u0022,\u0022key\u0022:\u0022feature.pull.request.suggestions\u0022,\u0022type\u0022:\u0022BOOLEAN\u0022}";
WRM._unparsedData["com.atlassian.bitbucket.server.config-wrm-data:content.upload.max.size.data"]="{\u0022value\u0022:\u00225242880\u0022,\u0022key\u0022:\u0022content.upload.max.size\u0022,\u0022type\u0022:\u0022NUMBER\u0022}";
WRM._unparsedData["com.atlassian.bitbucket.server.config-wrm-data:page.max.source.lines.data"]="{\u0022value\u0022:\u002220000\u0022,\u0022key\u0022:\u0022page.max.source.lines\u0022,\u0022type\u0022:\u0022NUMBER\u0022}";
WRM._unparsedData["com.atlassian.bitbucket.server.bitbucket-client-web-fragments:file-source-toolbar-primary-location._unused_"]="null";
WRM._unparsedData["com.atlassian.bitbucket.server.bitbucket-client-web-fragments:file-source-toolbar-secondary-location._unused_"]="null";
WRM._unparsedData["com.atlassian.bitbucket.server.bitbucket-client-web-fragments:file-diff-toolbar-primary-location._unused_"]="null";
WRM._unparsedData["com.atlassian.bitbucket.server.bitbucket-client-web-fragments:file-diff-toolbar-secondary-location._unused_"]="null";
WRM._unparsedData["com.atlassian.bitbucket.server.bitbucket-client-web-fragments:branch-layout-actions-dropdown-location._unused_"]="null";
WRM._unparsedData["com.atlassian.bitbucket.server.bitbucket-client-web-fragments:clone-dialog-options-location._unused_"]="null";
WRM._unparsedData["com.atlassian.bitbucket.server.bitbucket-mirroring-upstream:preferred-mirror.preferred-mirror-id"]="\u0022\u0022";
WRM._unparsedData["com.atlassian.analytics.analytics-client:policy-update-init.policy-update-data-provider"]="false";
WRM._unparsedData["com.atlassian.analytics.analytics-client:programmatic-analytics-init.programmatic-analytics-data-provider"]="false";
WRM._unparsedData["com.atlassian.plugins.atlassian-connect-plugin:dialog-options.data"]="{\u0022dialogOptions\u0022:{},\u0022inlineDialogOptions\u0022:{},\u0022dialogModules\u0022:{\u0022bitbucket.mirror.bo1u-4tsw-dpkf-jicq\u0022:{\u0022mirror-all-confirm\u0022:{\u0022url\u0022:\u0022/plugins/servlet/admin/mirror/dialog?id=mirror-all-confirm\u0022,\u0022options\u0022:{\u0022size\u0022:\u0022medium\u0022},\u0022key\u0022:\u0022mirror-all-confirm\u0022}}}}";
WRM._unparsedData["com.atlassian.bitbucket.server.feature-wrm-data:user.time.zone.onboarding.data"]="true";
if(window.WRM._dataArrived)window.WRM._dataArrived();</script>
<link type="text/css" rel="stylesheet" href="/s/70162e017675357530ff29dd4c9e9c31-CDN/-107657968/0d41a64/18/6ad39e6148758eeeb9bd29bc14a8c464/_/download/contextbatch/css/_super/batch.css" data-wrm-key="_super" data-wrm-batch-type="context" media="all">
<link type="text/css" rel="stylesheet" href="/s/7cd5de94ad2dbd39c2ed5d80d1c14a7a-CDN/-107657968/0d41a64/18/1c53f15bff01697cf847739f960f9eee/_/download/contextbatch/css/bitbucket.page.repository.fileContent,bitbucket.feature.files.fileHandlers,bitbucket.layout.files,bitbucket.layout.branch,bitbucket.layout.repository,atl.general,bitbucket.layout.base,bitbucket.layout.entity,-_super/batch.css?feature.smart.mirrors.enabled=true&amp;hasConnectAddons=true&amp;isJiraLinked=true" data-wrm-key="bitbucket.page.repository.fileContent,bitbucket.feature.files.fileHandlers,bitbucket.layout.files,bitbucket.layout.branch,bitbucket.layout.repository,atl.general,bitbucket.layout.base,bitbucket.layout.entity,-_super" data-wrm-batch-type="context" media="all">
<link type="text/css" rel="stylesheet" href="/s/d41d8cd98f00b204e9800998ecf8427e-T/-107657968/0d41a64/18/5.11.1/_/download/batch/com.nerdwin15.stash-stash-webhook-jenkins:repository-bitbucket-builds-page/com.nerdwin15.stash-stash-webhook-jenkins:repository-bitbucket-builds-page.css" data-wrm-key="com.nerdwin15.stash-stash-webhook-jenkins:repository-bitbucket-builds-page" data-wrm-batch-type="resource" media="all">
<script type="text/javascript" src="/s/387e1d5a1cb16b46dc55bd4e0c9238c0-CDN/-107657968/0d41a64/18/6ad39e6148758eeeb9bd29bc14a8c464/_/download/contextbatch/js/_super/batch.js?locale=en-US" data-wrm-key="_super" data-wrm-batch-type="context" data-initially-rendered></script>
<script type="text/javascript" src="/s/5a62b84244159f26ff51617db454b68d-CDN/-107657968/0d41a64/18/1c53f15bff01697cf847739f960f9eee/_/download/contextbatch/js/bitbucket.page.repository.fileContent,bitbucket.feature.files.fileHandlers,bitbucket.layout.files,bitbucket.layout.branch,bitbucket.layout.repository,atl.general,bitbucket.layout.base,bitbucket.layout.entity,-_super/batch.js?feature.smart.mirrors.enabled=true&amp;hasConnectAddons=true&amp;isJiraLinked=true&amp;locale=en-US" data-wrm-key="bitbucket.page.repository.fileContent,bitbucket.feature.files.fileHandlers,bitbucket.layout.files,bitbucket.layout.branch,bitbucket.layout.repository,atl.general,bitbucket.layout.base,bitbucket.layout.entity,-_super" data-wrm-batch-type="context" data-initially-rendered></script>
<script type="text/javascript" src="/s/d41d8cd98f00b204e9800998ecf8427e-T/-107657968/0d41a64/18/5.11.1/_/download/batch/com.nerdwin15.stash-stash-webhook-jenkins:repository-bitbucket-builds-page/com.nerdwin15.stash-stash-webhook-jenkins:repository-bitbucket-builds-page.js" data-wrm-key="com.nerdwin15.stash-stash-webhook-jenkins:repository-bitbucket-builds-page" data-wrm-batch-type="resource" data-initially-rendered></script>
<script>(function(loader) {loader.load('bitbucket.web.repository.clone.dialog.options', {"com.atlassian.bitbucket.server.bitbucket-mirroring-upstream:mirroring-clone-urls":{"serverCondition":true}});loader.load('bitbucket.file-content.source.toolbar.primary', {});loader.load('bitbucket.file-content.diff.toolbar.secondary', {});loader.load('bitbucket.file-content.diff-view.options', {});loader.load('bitbucket.comments.info', {});loader.load('bitbucket.file-content.diff.toolbar.primary', {});loader.load('bitbucket.comments.extra', {"com.atlassian.bitbucket.server.bitbucket-jira:comment-issue-list":{"serverCondition":true}});loader.load('bitbucket.file-content.source.toolbar.secondary', {"com.atlassian.bitbucket.server.bitbucket-client-web-fragments:source-file-edit":{"serverCondition":true},"com.atlassian.bitbucket.server.bitbucket-git-lfs:source-file-lock":{"serverCondition":true}});loader.load('bitbucket.comments.actions', {"com.atlassian.bitbucket.server.bitbucket-jira:comment-create-issue-link":{"serverCondition":true}});loader.load('bitbucket.layout.repository', {"com.atlassian.bitbucket.server.bitbucket-repository-shortcuts:repository-shortcuts-url-scheme-whitelist-provider":{"urlSchemeWhitelist":["http://","https://","ftp://","ftps://","mailto:","skype:","callto:","facetime:","git:","irc:","irc6:","news:","nntp:","feed:","cvs:","svn:","mvn:","ssh:","itms:","notes:","smb:","hipchat://","sourcetree:","urn:","tel:","xmpp:","telnet:","vnc:","rdp:","whatsapp:","slack:","sip:","sips:","magnet:"]},"com.atlassian.bitbucket.server.bitbucket-page-data:markup-extension-provider":{"extensions":["md","markdown","mdown","mkdn","mkd","txt","text",""],"extensionsRaw":["txt","text",""],"name":"README"}});loader.load('bitbucket.branch.layout.actions.dropdown', {"com.atlassian.bitbucket.server.bitbucket-compare:compare-branch-action":{"serverCondition":true},"com.atlassian.bitbucket.server.bitbucket-sourcetree:sourcetree-checkout-action-branch-layout":{"serverCondition":true},"com.atlassian.bitbucket.server.bitbucket-client-web-fragments:download-branch-action":{"serverCondition":true},"com.atlassian.bitbucket.server.bitbucket-branch:create-branch-action":{"serverCondition":true}});}(_PageDataPlugin));</script><meta name="application-name" content="Bitbucket"><link rel="shortcut icon" type="image/x-icon" href="/s/-107657968/0d41a64/18/1.0/_/download/resources/com.atlassian.bitbucket.server.bitbucket-webpack-INTERNAL:favicon/favicon.ico" /><link rel="search" href="https://sourcecode.socialcoding.bosch.com/plugins/servlet/opensearch-descriptor" type="application/opensearchdescription+xml" title="Bitbucket code search"/></head><body class="aui-page-sidebar bitbucket-theme"><ul id="assistive-skip-links" class="assistive"><li><a href="#aui-sidebar-content">Skip to sidebar navigation</a></li><li><a href="#aui-page-panel-content-body">Skip to content</a></li></ul><div id="page"><!-- start #header --><header id="header" role="banner"><section class="notifications"><!--googleoff: all-->
<div id="wittified-banner-display">

        <style>
            .wittified-banner::before
            {
            content:"" !important;
            }
            .wittified-ann-banner-hide{
                display:none;
            }
            .wittified-ann-banner-show{
                display:block;
            }
            .explainer-icon {
                margin-top:-6px !important;
            }
        </style>
    <script> var wittified_countdown_tracker = {};</script>


    <script type="text/javascript">
        // common methods for all notifications

        var _MS_PER_MIN = 1000 * 60;

        function getMutePeriod(current, past) {
            return Math.floor((current - past) / _MS_PER_MIN);
        }

        function isAnnouncementMuted(notification) {
            var x = true;
            var user = AJS.params.remoteUser;
            var cookieValue = getCookie(cookieName);
            if (cookieValue) {
                var cookieValues = cookieValue.split(",");
                //console.log({cookieValues});
                var lastMuted = new Date().getTime();
                var mutedOnce = false;

                AJS.$.each(cookieValues, function (index, value) {
                    var notificationId = value.split("#")[0];
                    if (notificationId == notification.id) {
                        mutedOnce = true;
                        lastMuted = value.split("#")[1];
                        //configured mute value in global configuration.
                        var configuredMuteValue = 240;
                        var mutePeriod = getMutePeriod(new Date().getTime(), lastMuted);
                        if (mutePeriod < configuredMuteValue) {
                            x = false
                        } else {
                            x = true
                        }
                    }
                });
                if (!mutedOnce) {
                    console.log("===== cookie found but no entry");
                    x = true
                }
            } else {
                console.log("==== cookie not found ");
            }

            return x;
        }

        function setCookie(cookieName, cookieValue, options) {
            if(options){
                const expires = options.expires;
                document.cookie = [
                    cookieName, '=', cookieValue, ';', expires, ';path=/', ';SameSite=Strict;', (location.protocol === "https:" ? "Secure;" : "")
                ].join('')
            }else{
                document.cookie = cookieName + '=;path=/';
                document.cookie = [
                    cookieName, '=', cookieValue.toString(), ';path=/', ';SameSite=Strict;', (location.protocol === "https:" ? "Secure;" : "")
                ].join('')
            }
        }

        function getCookie(cname) {
            var name = cname + "=";
            var decodedCookie = decodeURIComponent(document.cookie);
            var ca = decodedCookie.split(';');
            for(var i = 0; i <ca.length; i++) {
                var c = ca[i];
                while (c.charAt(0) == ' ') {
                    c = c.substring(1);
                }
                if (c.indexOf(name) == 0) {
                    return c.substring(name.length, c.length);
                }
            }
            return "";
        }

        function getCookieKey(userKey) {
            var salt = "WitTi@!",
                    keyLength = 15;
            var value = "";
            if (userKey != "") {
                var md = window.wittified.forge.create();
                md.update(userKey + salt);
                value = md.digest().toHex().substring(0, keyLength);
            } else {
                var anonCookieValue = getCookie('announcer-anon-cookie-id');
                if (anonCookieValue == undefined || anonCookieValue == '') {
                    var uuid = Math.random().toString(16).split('.')[1];
                    var md = window.wittified.forge.create();
                    md.update(uuid + salt);
                    value = md.digest().toHex().substring(0, keyLength);
                    setCookie('announcer-anon-cookie-id', value, false);
                } else {
                    value = anonCookieValue;
                }
            }
            return 'com.wittified.mute_' + value;
        }

        				var notifications = []
				var user = AJS.params.remoteUser;
		        var cookieName = getCookieKey(user);
		        var cookieValue = getCookie(cookieName);
        
    </script>

                

                    <div id="wittified-notification-863">
                                

                                                                
                <div class="aui-message aui-message-info wittified-banner" style="margin:0px;">
                    <div class="aui-group aui-group-split">
                    	<div class="aui-item" style="width:80%;">
					        <p style="margin-top:0px;"><strong>Bitbucket Platform Security upgrade to LTS version 7.6.16 successfully done</strong></p>
		                    		                        Dear developers, 
<br> 
<p>security update of Bitbucket to version 7.6.16 finished successfully</p>
		                    					    </div>
					    <div class="aui-item">
					                                    	<button class="aui-button aui-button-primary
		                            		                            "
	                                	                                    id="wittified-banner-submit-button-863" onclick="wittified_announcer_banner_dismiss(863, '', '');">
	                                Dismiss
                            	</button>
                            	                        	                        	<div id="wittified-notification-state-863"></div>
					    </div>

					</div>
                </div>
            </div>
                  	        <script>

            var user = AJS.params.remoteUser;
            var cookieName = getCookieKey(user);
            var cookieValue = getCookie(cookieName);
		    var mutedAnnouncements = [];
            var muteIndex = "";
            var cookieValues = cookieValue ? cookieValue.split(",") : [];
            var lastMuted = new Date().getTime();
            var mutedOnce = false;

            AJS.$.each(cookieValues, function (index, value) {
                var notificationId = value.split("#")[0];
                if (notificationId == 863) {
                    mutedOnce = true;
                    lastMuted = value.split("#")[1];
                    //configured mute value in global configuration.
                    var configuredMuteValue = 240;
                    var mutePeriod = getMutePeriod(new Date().getTime(), lastMuted);
                    if (mutePeriod < configuredMuteValue) {
                        // need to look for othe.
                        muteIndex = mutedAnnouncements.indexOf(notificationId);
                        if (muteIndex < 0) {
                            mutedAnnouncements.push(863);
                            AJS.$('#wittified-notification-863').addClass('wittified-ann-banner-hide');
                            AJS.$('#wittified-notification-863').removeClass('wittified-ann-banner-show');
                                                        }
                    } else {
                        console.log("Cookie found and entry time exceeded");
                        AJS.$('#wittified-notification-863').removeClass('wittified-ann-banner-hide');
                        AJS.$('#wittified-notification-863').addClass('wittified-ann-banner-show');
                                                if (muteIndex > 0) {
                            mutedAnnouncements.splice(muteIndex, 1);
                        }
                    }
                }
            });
            if (!mutedOnce) {
                console.log("Cookie found but entry doesn't");
                    AJS.$('#wittified-notification-863').removeClass('wittified-ann-banner-hide');
                    AJS.$('#wittified-notification-863').addClass('wittified-ann-banner-show');
                                if (muteIndex > 0) {
                    mutedAnnouncements.splice(muteIndex, 1);
                }
            }

		</script>
	      




<!-- FIXME: Use the rest interface for this -->
<script type="text/javascript">
	wittified_announcer_dialog_countdown_time = 5;
	function wittified_announcer_dialog_countdown()
	{

       var scheduleAgain = false;

        AJS.$('.wittified-dialog-button-countdown').each(function( index)
	    {
            var btn = AJS.$(this);
            var btnObj = AJS.$(this);

            if(btn)
            {
                if(btn.length) { btn = btn.get(0);}
                var id = btn.id
                if(id.indexOf('wittified-banner-submit-button-')>-1)
                {
                    id = id.substring('wittified-banner-submit-button-'.length);
                }
                if(id.indexOf('dialog-countdown-')>-1)
                {
                    id = id.substring('dialog-countdown-'.length);
                }
                if(AJS.$(this).data('notificationid'))
                {
                    id = AJS.$(this).data('notificationid');
                }




                if(wittified_countdown_tracker['notification-'+id]>-1)
                {
                    var wittified_announcer_dialog_countdown_time = wittified_countdown_tracker['notification-'+id];

                    var txt = btn.innerHTML;

                    var countDownIndex = txt.indexOf(' ( '+(wittified_announcer_dialog_countdown_time+1) +' )');
                    if(countDownIndex>0)
                    {
                        txt = txt.substring(0,countDownIndex);

                        if( wittified_announcer_dialog_countdown_time>0)
                        {
                        	txt = txt+' ( '+wittified_announcer_dialog_countdown_time+ ' )'

                        	if(false)
                        	{
                        		btn.setAttribute('aria-disabled','true');
                        	}
                        	else
                        	{
                        		btn.setAttribute('disabled','disabled');
                        	}
                        	btn.innerHTML = txt;
                        }
                        else
                        {
                            btn.innerHTML = txt;

                            if(false)
                     	   	{
                                btnObj.removeAttr('aria-disabled','true');
                                btnObj.attr('aria-disabled','false');
                              	btnObj.removeAttr('disabled','disabled');
                     	   	}
                            else
                            {
                            	btnObj.removeAttr('disabled','disabled');
                         	}
                        }
                    }
                    else
                    {
                    	txt = txt+' ( '+ wittified_announcer_dialog_countdown_time+' )';
                    	btn.innerHTML = txt;
                    	if(false)
                    	{
                    		btn.setAttribute('aria-disabled','true');
                    	}
                    	else
                    	{
                    		btn.setAttribute('disabled','disabled');
                    	}
                    }
                    wittified_announcer_dialog_countdown_time--;
                    wittified_countdown_tracker['notification-'+id] = wittified_announcer_dialog_countdown_time;
                    if( wittified_announcer_dialog_countdown_time>-1)
                    {
                        scheduleAgain = true;
                    }

                }
            }
        });
        if( scheduleAgain)
        {
            window.setTimeout( wittified_announcer_dialog_countdown, 1000);

        }
	}



    function wittified_announcer_banner_remindmelater(bannerID){
        AJS.$('#wittified-notification-'+bannerID).html('');
        var cookieName = getCookieKey(AJS.params.remoteUser);

        var cookieValue = getCookie(cookieName);
        if (cookieValue) {
                // need to check for 256 charectors
                if (cookieValue.length >= 200) {
                    cookieValue = cookieValue.substring(cookieValue.indexOf(",") + 1, cookieValue.length);
                }
                var cookieValues = cookieValue.split(",");
                var onceMuted = false;
                AJS.$.each(cookieValues, function (index, value) {
                    if (value.split("#")[0] == bannerID) {
                        // update the mute time.
                        // need to take care of cookie length.
                        cookieValues[index] = bannerID + "#" + new Date().getTime();
                        onceMuted = true;
                        return;
                    }
                });
                // not muted till now
                if (!onceMuted) {
                    // need to take care of cookie length.
                    cookieValues.push(bannerID + "#" + new Date().getTime());
                }
                setCookie(cookieName, cookieValues, false);
        } else {
            var cookieItems = [];
            cookieItems.push(bannerID + "#" + new Date().getTime());
            setCookie(cookieName, cookieItems, false);
        }
    }

    function reloadWindow(){

        window.location.reload();
    }

	function wittified_announcer_banner_dismiss( id,area, cb )
	{
		if(AJS.$('#dialog-countdown-'+id).attr('aria-disabled') == 'true'){
            return;
        }
        if(false) {
            AJS.dialog2('#announcer-confirmation-dialog-'+id).remove();
        }

		var bannerID = id;
		if(!area) { area = 'wittified-banner-submit-button';}
        AJS.$('#'+ area +'-'+id).html('<span class="aui-icon aui-icon-wait">Wait</span>');

		AJS.$.ajax(
		 {
		    "contentType": "application/json",
		    "data": JSON.stringify( { "loadTime": "1660681929508", "id":id, "key": "$currentKey" }),
		    "url":  AJS.contextPath()+'/rest/announcer/1.0/accept/accept',
              "type": "POST",
		    "method":"POST"
		    }).always(function(state)
		    {
                if(state.status=='success')
                {
                    AJS.$('#wittified-notification-'+bannerID).html('');
                    if(cb)
                    {cb();}
                }
                else
                {
                    AJS.$('#wittified-notification-state-'+bannerID).html('Submission error, please reload the page');
                }
		});
	}
</script>


    <script type="text/javascript">
console.log("display flag" + true);

function wittified_announcer_flag_remindmelater(bannerID){
        window.wittifiedAllFlags[bannerID+''].close();
        var cookieName = getCookieKey(AJS.params.remoteUser);
        var cookieValue = getCookie(cookieName);

        if (cookieValue) {
            // need to check for 256 charectors
            if (cookieValue.length >= 200) {
                cookieValue = cookieValue.substring(cookieValue.indexOf(",") + 1, cookieValue.length);
            }
            var cookieValues = cookieValue.split(",");
            var onceMuted = false;
            AJS.$.each(cookieValues, function (index, value) {
                if (value.split("#")[0] == bannerID) {
                    // update the mute time.
                    // need to take care of cookie length.
                    cookieValues[index] = bannerID + "#" + new Date().getTime();
                    onceMuted = true;
                    return;
                }
            });
            // not muted till now
            if (!onceMuted) {
                // need to take care of cookie length.
                cookieValues.push(bannerID + "#" + new Date().getTime());
            }
            setCookie(cookieName, cookieValues, false);
        } else {
            var cookieItems = [];
            cookieItems.push(bannerID + "#" + new Date().getTime());
            setCookie(cookieName, cookieItems, false);
        }
    }

	function wittified_announcer_flag_dismiss(event,id )
	{
	    event.preventDefault();


	     AJS.$('#wittified-notification-flag-'+id).html('<span class="aui-icon aui-icon-wait">Wait</span>');



		AJS.$.ajax(
		 {
		    "contentType": "application/json",
		    "data": JSON.stringify( { "loadTime": "1660681929508", "id":id, "key": "$currentKey" }),
		    "url":  AJS.contextPath()+'/rest/announcer/1.0/accept/accept',
		    "type": "POST",
		    "method":"POST"
		    })
  .always(function(state)
		    {
                if(state.status=='success')
                {
    			    window.wittifiedAllFlags[id+''].close();

                }
                else
                {
                    AJS.$('#wittified-notification-state-'+bannerID).html('Submission error, please reload the page');
                }
		});
	}




	function wittified_render_announcer_flags()
	{
        var _notifications = [];
        var x = true;
				    AJS.$.get( AJS.contextPath() + '/rest/announcer/1.0/accept/flags', function (notifications)
				{
			if(notifications)
			{
			    require(['aui/flag'], function(flag)
			    {

	                var gotDelayedFlags = false;
                    for(var i=0;i<notifications.length;i++)
                    {

                        var notification = notifications[i];
                        if((notification.type=='flag') &&  (AJS.$('#wittified-flag-announcer-'+notification.id).length==0))
                        {
                            var btn = '';

                            if(notification.showButton)
                            {

                                var buttonText = notification.buttonText;
                                if(!buttonText || (buttonText=='')|| (buttonText=='null')) { buttonText = "Dismiss"; }
                                var delayCss = '';
                                var disabled = '';
                                if(notification.delayButton)
                                {
                                    delayCss = ' wittified-dialog-button-countdown';
                                    disabled = ' disabled="true" '
                                    gotDelayedFlags = true;
                                }
                                btn ='<ul class="aui-nav-actions-list">' +
                '<li><button data-notificationid="'+notification.id+'" class="aui-button aui-button-primary '+delayCss+'" '+disabled+' onclick="wittified_announcer_flag_dismiss(event,'+notification.id+')" data-id="'+notification.id+'" id="wittified-notification-flag-'+notification.id+'">'+buttonText+'</button>'+
                '</li></ul>';
                            }
                                var body = '<div id="wittified-flag-announcer-'+notification.id+'" class="wittified-flag-dialog '+notification.cssClass+'">'+notification.contents +'<div id="wittified-notification-state-'+notification.id+'">'+btn+'</div></div>';

                            if(isAnnouncementMuted(notification)){
                                window.wittifiedAllFlags[notification.id + ''] = flag(
                                        {
                                            type: notification.display,
                                            title: notification.title,
                                            close: 'never',
                                            body: body
                                        });
                            }

                        }

                    }

                    if(gotDelayedFlags)
                    {
                        wittified_announcer_dialog_countdown_time = 5;
                        wittified_announcer_dialog_countdown();
                    }

			    });


			}
		}    , 'json' );

	}

            window.wittifiedAllFlags = {};
        AJS.toInit( function()
        {
    
        wittified_render_announcer_flags();
            });
    	</script>



<script>
                        AJS.toInit( function()
            {
                if( window.JIRA)
                {
                    JIRA.bind( JIRA.Events.ISSUE_REFRESHED, function( e, issueId, reason)
                    {
                        AJS.$.get( AJS.contextPath()+'/rest/announcer/1.0/jira/'+ issueId, function(html)
                        {

                            AJS.$('#wittified-banner-display').html( html );
                        });
                    });
                }
            });
                    </script>

		<script>
                            AJS.toInit( function() { window.setTimeout(wittified_announcer_dialog_countdown, 10)});
            
		</script>
		</div>
<!--googleon: all-->
</section><nav class="aui-header aui-dropdown2-trigger-group" role="navigation"><div class="aui-header-inner"><div class="aui-header-before"><a class=" aui-dropdown2-trigger app-switcher-trigger" aria-controls="app-switcher" aria-haspopup="true" role="button" tabindex="0" data-aui-trigger href="#app-switcher"><span class="aui-icon aui-icon-small aui-iconfont-appswitcher">Linked Applications</span></a><div id="app-switcher" class="aui-dropdown2 aui-style-default" role="menu" aria-hidden="true" data-is-switcher="true" data-environment="{&quot;isUserAdmin&quot;:false,&quot;isAppSuggestionAvailable&quot;:false,&quot;isSiteAdminUser&quot;:false}"><div role="application"><div class="app-switcher-loading">Loading&hellip;</div></div></div></div><div class="aui-header-primary"><h1 id="logo" class="bitbucket-header-logo"><a href="https://sourcecode.socialcoding.bosch.com">Bitbucket</a></h1><ul class="aui-nav"><li class=" projects-link"><a href="/projects" class="projects-link" data-web-item-key="com.atlassian.bitbucket.server.bitbucket-server-web-fragments:projects-menu">Projects</a></li><li class="selected recent-repositories"><a id="repositories-menu-trigger"  class=" aui-dropdown2-trigger" aria-controls="com.atlassian.bitbucket.server.bitbucket-server-web-fragments-repositories-menu" aria-haspopup="true" role="button" tabindex="0" data-aui-trigger>Repositories</a><div id="com.atlassian.bitbucket.server.bitbucket-server-web-fragments-repositories-menu" class="aui-dropdown2 aui-style-default" role="menu" aria-hidden="true"><div role="application"><div class="aui-dropdown2-section recent-repositories-section"><strong role="presentation" class="aui-dropdown2-heading">Recently viewed</strong><ul class="aui-list-truncate" role="presentation"></ul></div></div></div></li><li class=""><a id="snippets-menu-trigger"  class=" aui-dropdown2-trigger" aria-controls="com.simplenia.stash.plugins.snippets-snippets-web-item" aria-haspopup="true" role="button" tabindex="0" data-aui-trigger>Snippets</a><div id="com.simplenia.stash.plugins.snippets-snippets-web-item" class="aui-dropdown2 aui-style-default" role="menu" aria-hidden="true"><div role="application"><div class="aui-dropdown2-section snippets-links-web-section"><ul class="aui-list-truncate" role="presentation"><li role="presentation"><a href="/snippets" data-web-item-key="com.simplenia.stash.plugins.snippets:snippets-links-my-snippets-web-item">My snippets</a></li><li role="presentation"><a href="/snippets/starred" data-web-item-key="com.simplenia.stash.plugins.snippets:snippets-links-starred-snippets-web-item">Starred</a></li><li role="presentation"><a href="/snippets/watched" data-web-item-key="com.simplenia.stash.plugins.snippets:snippets-links-watched-snippets-web-item">Watched</a></li><li role="presentation"><a href="/snippets/browse" data-web-item-key="com.simplenia.stash.plugins.snippets:snippets-links-browse-snippets-web-item">Browse</a></li></ul></div><div class="aui-dropdown2-section snippets-actions-web-section"><ul class="aui-list-truncate" role="presentation"><li role="presentation"><a href="/snippets/create" data-web-item-key="com.simplenia.stash.plugins.snippets:snippets-actions-create-snippet-web-item">Create snippet</a></li></ul></div></div></div></li></ul></div><div class="aui-header-secondary"><ul class="aui-nav"><li><div id="quick-search-loader"></div><script>jQuery(document).ready(function () {require(['bitbucket-plugin-search/internal/component/quick-search/quick-search-loader'], function (loader) {loader.onReady('#quick-search-loader');}) ;}) ;</script></li><li class=" help-link"title="Help"><a class=" aui-dropdown2-trigger aui-dropdown2-trigger-arrowless" aria-controls="com.atlassian.bitbucket.server.bitbucket-server-web-fragments-help-menu" aria-haspopup="true" role="button" tabindex="0" data-aui-trigger><span class="aui-icon aui-icon-small aui-icon-small aui-iconfont-question-circle">Help</span></a><div id="com.atlassian.bitbucket.server.bitbucket-server-web-fragments-help-menu" class="aui-dropdown2 aui-style-default" role="menu" aria-hidden="true"><div role="application"><div class="aui-dropdown2-section help-items-section"><ul class="aui-list-truncate" role="presentation"><li role="presentation"><a href="https://confluence.atlassian.com/display/BITBUCKETSERVER076/Bitbucket+Data+Center+and+Server+documentation?utm_campaign=in-app-help&amp;amp;utm_medium=in-app-help&amp;amp;utm_source=stash" title="Go to the online documentation for Bitbucket" data-web-item-key="com.atlassian.bitbucket.server.bitbucket-server-web-fragments:general-help">Online help</a></li><li role="presentation"><a href="https://www.atlassian.com/git?utm_campaign=learn-git&amp;utm_medium=in-app-help&amp;utm_source=stash" title="Learn about Git commands &amp; workflows" data-web-item-key="com.atlassian.bitbucket.server.bitbucket-server-web-fragments:learn-git">Learn Git</a></li><li role="presentation"><a href="/getting-started" class="getting-started-page-link" title="Overview of Bitbucket features" data-web-item-key="com.atlassian.bitbucket.server.bitbucket-server-web-fragments:getting-started-page-help-link">Welcome to Bitbucket</a></li><li role="presentation"><a href="/#" class="keyboard-shortcut-link" title="Discover keyboard shortcuts in Bitbucket" data-web-item-key="com.atlassian.bitbucket.server.bitbucket-server-web-fragments:keyboard-shortcuts-help-link">Keyboard shortcuts</a></li><li role="presentation"><a href="https://go.atlassian.com/bitbucket-server-whats-new?utm_campaign=in-app-help&amp;utm_medium=in-app-help&amp;utm_source=stash" title="Learn about what&#39;s new in Bitbucket" data-web-item-key="com.atlassian.bitbucket.server.bitbucket-server-web-fragments:whats-new-link">What&#39;s new</a></li><li role="presentation"><a href="https://go.atlassian.com/bitbucket-server-community?utm_campaign=in-app-help&amp;utm_medium=in-app-help&amp;utm_source=stash" title="Explore the Atlassian community" data-web-item-key="com.atlassian.bitbucket.server.bitbucket-server-web-fragments:community-link">Community</a></li></ul></div></div></div></li><li class=" alerts-menu"title="View system alerts"><a href="#alerts" id="alerts-trigger" class="alerts-menu" title="View system alerts" data-web-item-key="com.atlassian.bitbucket.server.bitbucket-server-web-fragments:global-alerts-menu-item">Alerts</a></li><li class="inbox-menu" title="View your unapproved pull requests"><a href="#inbox" id="inbox-trigger" data-aui-trigger aria-controls="inline-dialog-inbox-pull-requests-content" aria-label="View your unapproved pull requests" ><span class="aui-icon aui-icon-small aui-iconfont-tray">Inbox</span></a><aui-inline-dialog id="inline-dialog-inbox-pull-requests-content" alignment="bottom right"><aui-spinner size="medium" /></aui-inline-dialog></li><li class="user-dropdown"><a class=" aui-dropdown2-trigger user-dropdown-trigger aui-dropdown2-trigger-arrowless" aria-controls="user-dropdown-menu" aria-haspopup="true" role="button" title="Logged in as Behrmann Nadine (CR/AIR2.1) (BEN2RNG)" data-container=".aui-header-secondary" tabindex="0" data-aui-trigger><span id="current-user" class="aui-avatar aui-avatar-small" data-emailaddress="Nadine.Behrmann@de.bosch.com" data-username="BEN2RNG" data-avatarurl-small="https://secure.gravatar.com/avatar/a7d5c0b84ef2b629735004700a6ebe0c.jpg?s=48&amp;d=mm"><span class="aui-avatar-inner"><img src="https://secure.gravatar.com/avatar/a7d5c0b84ef2b629735004700a6ebe0c.jpg?s=48&amp;d=mm" alt="Logged in as Behrmann Nadine (CR/AIR2.1) (BEN2RNG)" /></span></span></a><div id="user-dropdown-menu" class="aui-dropdown2 aui-style-default" role="menu" aria-hidden="true"><div role="application"><div class="aui-dropdown2-section user-settings-section"><ul class="aui-list-truncate" role="presentation"><li role="presentation"><a href="/profile" data-web-item-key="com.atlassian.bitbucket.server.bitbucket-server-web-fragments:profile-menu-item">View profile</a></li><li role="presentation"><a href="/account" data-web-item-key="com.atlassian.bitbucket.server.bitbucket-server-web-fragments:account-menu-item">Manage account</a></li><li role="presentation"><a href="/snippets" data-web-item-key="com.simplenia.stash.plugins.snippets:user-settings-my-snippets-web-item">My snippets</a></li><li role="presentation"><a href="/plugins/servlet/wittified/profile/notifications" id="announcer-usermenu" data-web-item-key="com.wittified.atl-announcer-stash:stash-profile-user-menu">Acknowledged notifications</a></li></ul></div><div class="aui-dropdown2-section user-logout-section"><ul class="aui-list-truncate" role="presentation"><li role="presentation"><a href="/j_atl_security_logout" class="logout-link" data-web-item-key="com.atlassian.bitbucket.server.bitbucket-server-web-fragments:logout-menu-item">Log out</a></li></ul></div></div></div></li></ul></div></div> <!-- End .aui-header-inner --></nav> <!-- End .aui-header --></header><!-- End #header --><!-- Start #content --><section id="content" role="main" tabindex="-1" data-timezone="-120"  data-repoSlug="uvast_final" data-projectKey="~BEN2RNG" data-repoName="UVAST_final" data-projectName="Behrmann Nadine (CR/AIR2.1)"><section class="notifications"></section><div id="aui-sidebar-content" class="aui-sidebar "  tabindex="-1"><div class="aui-sidebar-wrapper"><div class="aui-sidebar-body"><script>require('bitbucket/internal/widget/sidebar/sidebar').preload();</script><header class="aui-page-header"><div class="aui-page-header-inner"><div class="aui-page-header-image"><a href="/users/ben2rng"><span class="aui-avatar aui-avatar-large" data-tooltip="Behrmann Nadine (CR/AIR2.1)"><span class="aui-avatar-inner"><img src="https://secure.gravatar.com/avatar/a7d5c0b84ef2b629735004700a6ebe0c.jpg?s=96&amp;d=mm" alt="Behrmann Nadine (CR/AIR2.1)" /></span></span></a></div><!-- .aui-page-header-image --><div class="aui-page-header-main entity-item"><ol class="aui-nav aui-nav-breadcrumbs"><li><a href="/users/ben2rng" title="Behrmann Nadine (CR/AIR2.1)">Behrmann Nadine (CR/AIR2.1)</a></li></ol><h1><span class="entity-name" title="UVAST_final">UVAST_final</span></h1><div></div></div><!-- .aui-page-header-main --></div><!-- .aui-page-header-inner --></header><!-- .aui-page-header --><nav class="aui-navgroup aui-navgroup-vertical" role="navigation"><div class="aui-navgroup-inner"><div class="aui-sidebar-group aui-sidebar-group-tier-one sidebar-actions"><div class="aui-nav-heading"><strong>Actions</strong></div><ul class="aui-nav"><li class=" clone-repo"><a href="#" class="aui-nav-item "  id=clone-repo-button data-web-item-key=com.atlassian.bitbucket.server.bitbucket-server-web-fragments:repository-clone  title=Clone this repository><span class="aui-icon icon-clone"></span><span class="aui-nav-item-label">Clone</span></a></li><li class=" create-branch"><a href="/plugins/servlet/create-branch" class="aui-nav-item "  data-web-item-key=com.atlassian.bitbucket.server.bitbucket-branch:create-branch-repository-action ><span class="aui-icon icon-create-branch"></span><span class="aui-nav-item-label">Create branch</span></a></li><li class=" create-pull-request"><a href="/users/ben2rng/repos/uvast_final/pull-requests?create" class="aui-nav-item "  data-web-item-key=com.atlassian.bitbucket.server.bitbucket-server-web-fragments:repository-pull-request  title=Create a new pull request><span class="aui-icon icon-create-pull-request"></span><span class="aui-nav-item-label">Create pull request</span></a></li><li><a href="/snippets/create" class="aui-nav-item "  data-web-item-key=com.simplenia.stash.plugins.snippets:add-snippet ><span class="aui-icon aui-icon icon-add-snippet"></span><span class="aui-nav-item-label">Create snippet</span></a></li><li><a href="/users/ben2rng/repos/uvast_final/compare" class="aui-nav-item "  id=repository-nav-compare data-web-item-key=com.atlassian.bitbucket.server.bitbucket-server-web-fragments:bitbucket.repository.nav.compare ><span class="aui-icon icon-compare"></span><span class="aui-nav-item-label">Compare</span></a></li></ul></div><aui-inline-dialog id="repo-clone-dialog" alignment="left top"><div id="clone-repo-dialog-content"><div class="clone-url"><div class="aui-buttons"><button id="http-clone-url" class="aui-button repository-protocol"  data-module-key="http-clone-url" data-clone-url="https://sourcecode.socialcoding.bosch.com/scm/~ben2rng/uvast_final.git" autocomplete="off" aria-disabled="true" disabled="disabled" >HTTP</button><input type="text" class="text quick-copy-text stash-text clone-url-input" readonly="readonly" spellcheck="false" value=""/></div><div id="clone-dialog-options"><!-- This is a client-web-panel --></div><div id="clone-dialog-help-info"><p><a target="_blank" href="https://www.atlassian.com/git/tutorials/setting-up-a-repository/git-clone?utm_campaign=learn-git-clone&amp;utm_medium=in-app-help&amp;utm_source=stash">Learn more about cloning repositories</a></p><p><div id="contributing-guidelines-clone-placeholder" class="hidden"></div></p></div></div><div class="sourcetree-panel"><a id="sourcetree-clone-button" class="aui-button aui-button-primary sourcetree-button"  href="sourcetree://cloneRepo/https://sourcecode.socialcoding.bosch.com/scm/~ben2rng/uvast_final.git" autocomplete="off" tabindex="0">Clone in Sourcetree</a><p><a href="https://www.sourcetreeapp.com" target="_blank">Sourcetree</a> is a free Git and Mercurial client for Windows and Mac.</p></div></div></aui-inline-dialog><div class="aui-sidebar-group aui-sidebar-group-tier-one sidebar-navigation"><div class="aui-nav-heading"><strong>Navigation</strong></div><ul class="aui-nav"><li class="aui-nav-selected"><a href="/users/ben2rng/repos/uvast_final/browse" class="aui-nav-item "  id=repository-nav-files data-web-item-key=com.atlassian.bitbucket.server.bitbucket-server-web-fragments:bitbucket.repository.nav.files ><span class="aui-icon icon-source"></span><span class="aui-nav-item-label">Source</span></a></li><li class=" commits-nav"><a href="/users/ben2rng/repos/uvast_final/commits" class="aui-nav-item "  id=repository-nav-commits data-web-item-key=com.atlassian.bitbucket.server.bitbucket-server-web-fragments:bitbucket.repository.nav.commits ><span class="aui-icon icon-commits"></span><span class="aui-nav-item-label">Commits</span></a></li><li><a href="/users/ben2rng/repos/uvast_final/branches" class="aui-nav-item "  id=repository-nav-branches data-web-item-key=com.atlassian.bitbucket.server.bitbucket-server-web-fragments:bitbucket.repository.nav.branches ><span class="aui-icon icon-branches"></span><span class="aui-nav-item-label">Branches</span></a></li><li><a href="/plugins/servlet/bb_ag/projects/~BEN2RNG/repos/uvast_final/commits" class="aui-nav-item "  data-web-item-key=com.bit-booster.graph:bb-graph ><span class="aui-icon aui-icon bb-graph-icon"></span><span class="aui-nav-item-label">All Branches Graph</span></a></li><li><a href="/users/ben2rng/repos/uvast_final/pull-requests" class="aui-nav-item "  id=repository-nav-pull-requests data-web-item-key=com.atlassian.bitbucket.server.bitbucket-server-web-fragments:bitbucket.repository.nav.pull-requests ><span class="aui-icon icon-pull-requests"></span> <span class="aui-nav-item-label">Pull requests</span></a></li><li class=" forks-nav"><a href="/users/ben2rng/repos/uvast_final/forks" class="aui-nav-item "  id=repository-nav-forks data-web-item-key=com.atlassian.bitbucket.server.bitbucket-server-web-fragments:bitbucket.repository.nav.forks ><span class="aui-icon icon-forks"></span><span class="aui-nav-item-label">Forks</span></a></li><li><a href="/projects/~BEN2RNG/repos/uvast_final/builds" class="aui-nav-item "  id=repository-nav-builds data-web-item-key=com.atlassian.bitbucket.server.bitbucket-server-web-fragments:bitbucket.repository.nav.builds ><span class="aui-icon icon-builds"></span> <aui-badge class=" nav-onboarding-badge">New</aui-badge><span class="aui-nav-item-label">Builds</span></a></li><li class=" show-bitbucket-builds"><button class="aui-nav-item aui-button aui-button-subtle "  data-web-item-key=com.nerdwin15.stash-stash-webhook-jenkins:repository-show-bitbucket-builds ><span class="aui-icon aui-icon-small aui-iconfont-cross"></span><span class="aui-nav-item-label">Builds</span></button></li></ul></div><div class="aui-sidebar-group aui-sidebar-group-tier-one sidebar-navigation extra-section section-featured-items" data-web-section-key="bitbucket.web.sidebar.repository.nav.shortcuts"><div class="aui-nav-heading" id="bitbucket.web.sidebar.repository.nav.shortcuts-heading"><strong>Shortcuts</strong></div><ul class="aui-nav" role="group" aria-labelledby="bitbucket.web.sidebar.repository.nav.shortcuts-heading"></ul></div><div class="aui-sidebar-group sidebar-navigation extra-section add-shortcut-panel" data-web-section-key="bitbucket.web.sidebar.repository.nav.shortcuts"><ul class="aui-nav"><li><button class="aui-nav-item aui-button aui-button-subtle "  id=add-repo-shortcut-trigger data-web-item-key=com.atlassian.bitbucket.server.bitbucket-repository-shortcuts:bitbucket.web.sidebar.repository.nav.add.shortcut ><span class="aui-icon icon-add"></span><span class="aui-nav-item-label">Add shortcut</span></button></li></ul></div><div class="aui-sidebar-group aui-sidebar-group-tier-one sidebar-settings-group"><div class="aui-nav-heading"><strong> </strong></div><ul class="aui-nav"><li class=" aui-sidebar-settings-button"><a href="/users/ben2rng/repos/uvast_final/settings" class="aui-nav-item "  data-web-item-key=com.atlassian.bitbucket.server.bitbucket-server-web-fragments:bitbucket.repository.nav.settings ><span class="aui-icon icon-settings"></span><span class="aui-nav-item-label">Repository settings</span></a></li></ul></div></div></nav></div><div class="aui-sidebar-footer"><a class="aui-button aui-button-subtle aui-sidebar-toggle aui-sidebar-footer-tipsy" data-tooltip="Expand sidebar ( [ )" href="#"><span class="aui-icon aui-icon-small"></span></a></div></div></div><div class="aui-page-panel content-body" id="aui-page-panel-content-body" tabindex="-1"><div class="aui-page-panel-inner"><section class="aui-page-panel-content"><div id="default-reviewers-feature-discovery-meta" data-feature-discovery-level="repo-admin" ></div><header class="aui-page-header page-header-flex"><div class="aui-page-header-inner"><div class="aui-page-header-main"><ol class="aui-nav aui-nav-breadcrumbs repository-breadcrumbs"><li><a href="/users/ben2rng" title="Behrmann Nadine (CR/AIR2.1)">Behrmann Nadine (CR/AIR2.1)</a></li><li class="aui-nav-selected"><a href="/users/ben2rng/repos/uvast_final/browse" title="UVAST_final">UVAST_final</a></li></ol><h2 class="page-panel-content-header">Source</h2></div><!-- .aui-page-header-main --></div><!-- .aui-page-header-inner --></header><!-- .aui-page-header --><div class="aui-toolbar2 branch-selector-toolbar" role="toolbar"><div class="aui-toolbar2-inner"><div class="aui-toolbar2-primary"><div class="aui-group"><div class="aui-item"><div class="aui-buttons"><button type="button" id="repository-layout-revision-selector" data-aui-trigger aria-controls="inline-dialog-repository-layout-revision-selector-dialog" class="aui-button searchable-selector-trigger revision-reference-selector-trigger" title="master"><span class="aui-icon aui-icon-small aui-iconfont-branch">Branch</span><span class="name" title="master" data-id="refs/heads/master" data-revision-ref="{&quot;latestCommit&quot;:&quot;c87268b263fade848281740fd7fd2bbde4192b7f&quot;,&quot;isDefault&quot;:true,&quot;id&quot;:&quot;refs/heads/master&quot;,&quot;displayId&quot;:&quot;master&quot;,&quot;type&quot;:{&quot;name&quot;:&quot;Branch&quot;,&quot;id&quot;:&quot;branch&quot;}}">master</span></button><aui-inline-dialog id="inline-dialog-repository-layout-revision-selector-dialog" class="searchable-selector-dialog" alignment="bottom left" alignment-static></aui-inline-dialog><button id="branch-actions"  class=" aui-dropdown2-trigger aui-button aui-dropdown2-trigger-arrowless" aria-controls="branch-actions-menu" aria-haspopup="true" role="button" data-aui-trigger autocomplete="off" type="button"><span class="aui-icon aui-icon-small aui-iconfont-more">Branch actions</span></button></div></div><div class="aui-item"><div class="breadcrumbs" ><span class="file-path"><a href="/users/ben2rng/repos/uvast_final/browse">UVAST_final</a></span><span class="sep">/</span><span class="stub">compute_mean_dur.py</span></div></div></div></div><div class="aui-toolbar2-secondary commit-badge-container"><div class="commit-badge-oneline"><div class="double-avatar-with-name avatar-with-name"><span class="aui-avatar aui-avatar-small user-avatar first-person" data-username="BEN2RNG"><span class="aui-avatar-inner"><img src="https://secure.gravatar.com/avatar/a7d5c0b84ef2b629735004700a6ebe0c.jpg?s=48&amp;d=mm" alt="Behrmann Nadine (CR/AIR2.1)" /></span></span></div><span class="commit-details"><a href="/users/ben2rng" class="commit-author"title="Behrmann Nadine (CR/AIR2.1)">Behrmann Nadine (CR/AIR2.1)</a> authored <a class="commitid" href="/users/ben2rng/repos/uvast_final/commits/c87268b263fade848281740fd7fd2bbde4192b7f" data-commit-message="add script to compute mean durations" data-commitid="c87268b263fade848281740fd7fd2bbde4192b7f">c87268b263f</a><time datetime="2022-08-16T22:28:20+0200" title="16 August 2022 10:28 PM">3 mins ago</time></span></div></div></div></div></section><!-- .aui-page-panel-content --></div><!-- .aui-page-panel-inner --></div><!-- .aui-page-panel --></section><!-- End #content --><!-- Start #footer --><footer id="footer" role="contentinfo"><section class="notifications"></section><section class="footer-body"><ul><li data-key="footer.license.message">Git repository management for enterprise teams powered by <a href="https://www.atlassian.com/software/bitbucket/">Atlassian Bitbucket</a></li></ul><ul><li>Atlassian Bitbucket <span title="0d41a649553f1b52de0116249e9ea158357443c2" id="product-version" data-commitid="0d41a649553f1b52de0116249e9ea158357443c2" data-system-build-number="0d41a64"> v7.6.16</span></li><li data-key="footer.links.documentation"><a href="https://confluence.atlassian.com/display/BITBUCKETSERVER076/Bitbucket+Data+Center+and+Server+documentation?utm_campaign=in-app-help&amp;utm_medium=in-app-help&amp;utm_source=stash" target="_blank">Documentation</a></li><li data-key="footer.links.jac"><a href="https://jira.atlassian.com/browse/BSERV?utm_campaign=in-app-help&amp;utm_medium=in-app-help&amp;utm_source=stash" target="_blank">Request a feature</a></li><li data-key="footer.links.about"><a href="/about">About</a></li><li data-key="footer.links.contact.atlassian"><a href="https://www.atlassian.com/company/contact?utm_campaign=in-app-help&amp;utm_medium=in-app-help&amp;utm_source=stash" target="_blank">Contact Atlassian</a></li></ul>Generated by fe0vm04961 (0e6b6d1b-76aa-45f3-9538-95d37967dceb). Cluster contains 5 nodes.<div id="footer-logo"><a href="https://www.atlassian.com/" target="_blank">Atlassian</a></div></section></footer><!-- End #footer --></div><script>require('bitbucket/internal/layout/base/base').onReady({id : 47858, active: true, name : "BEN2RNG", slug : "ben2rng", displayName : "Behrmann Nadine (CR\/AIR2.1)", avatarUrl : "https:\/\/secure.gravatar.com\/avatar\/a7d5c0b84ef2b629735004700a6ebe0c.jpg?s\x3d48\x26d\x3dmm", emailAddress : "Nadine.Behrmann@de.bosch.com", type : "NORMAL"}, "Social Coding" ); require('bitbucket/internal/widget/keyboard-shortcuts/keyboard-shortcuts').onReady();</script><script>require('bitbucket/internal/layout/repository/repository').onReady({"slug":"uvast_final","id":111746,"name":"UVAST_final","hierarchyId":"7fcc0c0023f1739af914","scmId":"git","state":"AVAILABLE","statusMessage":"Available","forkable":true,"project":{"key":"~BEN2RNG","id":37789,"name":"Behrmann Nadine (CR/AIR2.1)","type":"PERSONAL","owner":{"name":"BEN2RNG","emailAddress":"Nadine.Behrmann@de.bosch.com","id":47858,"displayName":"Behrmann Nadine (CR/AIR2.1)","active":true,"slug":"ben2rng","type":"NORMAL","links":{"self":[{"href":"https://sourcecode.socialcoding.bosch.com/users/ben2rng"}]},"avatarUrl":"https://secure.gravatar.com/avatar/a7d5c0b84ef2b629735004700a6ebe0c.jpg?s=64&d=mm"},"links":{"self":[{"href":"https://sourcecode.socialcoding.bosch.com/users/ben2rng"}]},"avatarUrl":"https://secure.gravatar.com/avatar/a7d5c0b84ef2b629735004700a6ebe0c.jpg?s=64&d=mm"},"public":false,"links":{"clone":[{"href":"https://sourcecode.socialcoding.bosch.com/scm/~ben2rng/uvast_final.git","name":"http"},{"href":"ssh://git@sourcecode.socialcoding.bosch.com:7999/~ben2rng/uvast_final.git","name":"ssh"}],"self":[{"href":"https://sourcecode.socialcoding.bosch.com/users/ben2rng/repos/uvast_final/browse"}]}}, '#clone-repo-button');</script><div id="branch-actions-menu" class="aui-dropdown2 aui-style-default" role="menu" aria-hidden="true"><div role="application"></div></div><script>require('bitbucket/internal/layout/branch/branch').onReady('#repository-layout-revision-selector');</script><script>require('bitbucket/internal/layout/files/files').onReady(["compute_mean_dur.py"],{"latestCommit":"c87268b263fade848281740fd7fd2bbde4192b7f","isDefault":true,"id":"refs/heads/master","displayId":"master","type":{"name":"Branch","id":"branch"}}, '.branch-selector-toolbar .breadcrumbs',false);</script><script>require('bitbucket/internal/page/source/source').onReady( "compute_mean_dur.py",{"latestCommit":"c87268b263fade848281740fd7fd2bbde4192b7f","isDefault":true,"id":"refs/heads/master","displayId":"master","type":{"name":"Branch","id":"branch"}},{"id":"c87268b263fade848281740fd7fd2bbde4192b7f","displayId":"c87268b263f","author":{"name":"BEN2RNG","emailAddress":"Nadine.Behrmann@de.bosch.com","id":47858,"displayName":"Behrmann Nadine (CR/AIR2.1)","active":true,"slug":"ben2rng","type":"NORMAL","links":{"self":[{"href":"https://sourcecode.socialcoding.bosch.com/users/ben2rng"}]},"avatarUrl":"https://secure.gravatar.com/avatar/a7d5c0b84ef2b629735004700a6ebe0c.jpg?s=48&d=mm"},"authorTimestamp":1660681700000,"committer":{"name":"BEN2RNG","emailAddress":"Nadine.Behrmann@de.bosch.com","id":47858,"displayName":"Behrmann Nadine (CR/AIR2.1)","active":true,"slug":"ben2rng","type":"NORMAL","links":{"self":[{"href":"https://sourcecode.socialcoding.bosch.com/users/ben2rng"}]},"avatarUrl":"https://secure.gravatar.com/avatar/a7d5c0b84ef2b629735004700a6ebe0c.jpg?s=48&d=mm"},"committerTimestamp":1660681700000,"message":"add script to compute mean durations","parents":[{"id":"b2b6b893f6286eaf18b7a643c6f18cf581f2259e","displayId":"b2b6b893f62"}],"properties":{"change":{"type":"ADD","path":"compute_mean_dur.py"}}}, "compute_mean_dur.py","source", '#content .aui-page-panel-content', 'file-content',10,true,false,null);</script><script type="text/javascript">require('bitbucket/internal/layout/base/menu/repositories/recent').initMenu('repositories-menu-trigger');</script></body></html>