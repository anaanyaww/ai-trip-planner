"""Create the plans table and its audit trail.

The baseline schema: everything the API queries about a plan. The workflow state
itself is held by the checkpointer and is not part of this schema.

Revision ID: c40d06330c6b
Revises: 
Create Date: 2026-08-25 23:01:55.975378

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'c40d06330c6b'
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table('plans',
    sa.Column('plan_id', sa.String(length=36), nullable=False),
    sa.Column('status', sa.Enum('pending', 'researching', 'planning', 'awaiting_review', 'revising', 'finalizing', 'completed', 'failed', 'cancelled', name='plan_status', native_enum=False, length=32), nullable=False),
    sa.Column('destination', sa.String(length=120), nullable=False),
    sa.Column('idempotency_key', sa.String(length=128), nullable=True),
    sa.Column('request', sa.JSON(), nullable=False),
    sa.Column('draft', sa.JSON(), nullable=True),
    sa.Column('final', sa.JSON(), nullable=True),
    sa.Column('revision_count', sa.Integer(), nullable=False),
    sa.Column('error', sa.Text(), nullable=True),
    sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
    sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False),
    sa.PrimaryKeyConstraint('plan_id'),
    sa.UniqueConstraint('idempotency_key')
    )
    with op.batch_alter_table('plans', schema=None) as batch_op:
        batch_op.create_index(batch_op.f('ix_plans_status'), ['status'], unique=False)
        batch_op.create_index('ix_plans_status_updated_at', ['status', 'updated_at'], unique=False)

    op.create_table('plan_events',
    sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
    sa.Column('plan_id', sa.String(length=36), nullable=False),
    sa.Column('event', sa.String(length=64), nullable=False),
    sa.Column('detail', sa.JSON(), nullable=False),
    sa.Column('recorded_at', sa.DateTime(timezone=True), nullable=False),
    sa.ForeignKeyConstraint(['plan_id'], ['plans.plan_id'], ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id')
    )
    with op.batch_alter_table('plan_events', schema=None) as batch_op:
        batch_op.create_index(batch_op.f('ix_plan_events_plan_id'), ['plan_id'], unique=False)



def downgrade() -> None:
    """Downgrade schema."""
    with op.batch_alter_table('plan_events', schema=None) as batch_op:
        batch_op.drop_index(batch_op.f('ix_plan_events_plan_id'))

    op.drop_table('plan_events')
    with op.batch_alter_table('plans', schema=None) as batch_op:
        batch_op.drop_index('ix_plans_status_updated_at')
        batch_op.drop_index(batch_op.f('ix_plans_status'))

    op.drop_table('plans')
