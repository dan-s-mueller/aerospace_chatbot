import resend
from logger import get_logger
from models import BrainSubscription, BrainSettings

from repository.brain import get_brain_details
from repository.brain_subscription import get_brain_url

logger = get_logger(__name__)


def resend_invitation_email(
    brain_subscription: BrainSubscription,
    inviter_email: str,
    origin: str = "https://www.quivr.app",
):
    brains_settings = BrainSettings()  # pyright: ignore reportPrivateUsage=none
    resend.api_key = brains_settings.resend_api_key

    brain_url = get_brain_url(origin, brain_subscription.brain_id)

    invitation_brain = get_brain_details(brain_subscription.brain_id)
    if invitation_brain is None:
        raise Exception("Brain not found")
    brain_name = invitation_brain.name

    html_body = f"""
    <p>Brain {brain_name} has been shared with you by {inviter_email}.</p>
    <p><a href='{brain_url}'>Click here</a> to access your brain.</p>
    """

    try:
        r = resend.Emails.send(
            {
                "from": brains_settings.resend_email_address,
                "to": brain_subscription.email,
                "subject": "Quivr - Brain Shared With You",
                "html": html_body,
            }
        )
        logger.info("Resend response", r)
    except Exception as e:
        logger.error(f"Error sending email: {e}")
        return

    return r
